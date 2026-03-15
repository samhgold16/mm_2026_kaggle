# important packages
import pandas as pd
import numpy as np
from pathlib import Path

# setting pathing
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# setting years for analysis
# TRAIN_SEASONS = list(range(2003, 2026))
# PRED_SEASON   = 2026 

# functions to process data and extract info
def _stack_games(df):
    """
    Convert winner/loser wide format into two rows per game (one per team).
    """
    # choosing major stats to consider
    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
 
    # extracting major stats for winning team
    winner = df[["Season", "WTeamID", "WScore", "LScore"] +
                [f"W{c}" for c in stat_cols] +
                ["LOR", "LFGM", "LFGA"]].copy()
    winner.columns = (["Season", "TeamID", "Score", "OppScore"] +
                      stat_cols + ["OppOR", "OppFGM", "OppFGA"])
    # setting binary win/loss column
    winner["Win"] = 1
 
    # extracting major stats for losing team
    loser = df[["Season", "LTeamID", "LScore", "WScore"] +
               [f"L{c}" for c in stat_cols] +
               ["WOR", "WFGM", "WFGA"]].copy()
    loser.columns = (["Season", "TeamID", "Score", "OppScore"] +
                     stat_cols + ["OppOR", "OppFGM", "OppFGA"])
    # setting binary win/loss column
    loser["Win"] = 0
 
    return pd.concat([winner, loser], ignore_index = True)

# function to extract more unique factors
def _four_factors(g, tall):
    """
    Dean Oliver's Four Factors (offensive side) for a group of games and defensive stats.
    """
    idx = g.index
    fga = tall.loc[idx, "FGA"].sum()
    fgm = tall.loc[idx, "FGM"].sum()
    fgm3 = tall.loc[idx, "FGM3"].sum()
    ftm = tall.loc[idx, "FTM"].sum()
    fta = tall.loc[idx, "FTA"].sum()
    to = tall.loc[idx, "TO"].sum()
    orb = tall.loc[idx, "OR"].sum()
    opp_or = tall.loc[idx, "OppOR"].sum()
    opp_fgm = tall.loc[idx, "OppFGM"].sum()
    opp_fga = tall.loc[idx, "OppFGA"].sum()
 
    # using basic stats to calculate interesting metrics
    # make NaN if no stats recorded
    efg = (fgm + 0.5 * fgm3) / fga  if fga > 0 else np.nan
    tov = to / (fga + 0.44 * fta + to) if (fga + 0.44*fta + to) > 0 else np.nan
    orb_r = orb / (orb + opp_or) if (orb + opp_or) > 0 else np.nan
    ftr = ftm / fga if fga   > 0 else np.nan
    def_efg = opp_fgm / opp_fga if opp_fga > 0 else np.nan
 
    return pd.Series({"eFG_pct": efg,
                     "TOV_pct": tov,
                     "ORB_pct": orb_r,
                     "FTR":     ftr,
                     "Def_eFG_pct": def_efg})

# main function to combine above internal functions
def build_box_features(gender = "M"):
    """
    Compute per-team, per-season stats from REGULAR SEASON games only.
    """

    path = DATA_DIR / f"{gender}RegularSeasonDetailedResults.csv"
    reg  = pd.read_csv(path)
    tall = _stack_games(reg)
 
    grp = tall.groupby(["Season", "TeamID"])
 
    agg = grp.agg(PointsFor = ("Score", "mean"),
                  PointsAgn = ("OppScore", "mean"),
                  WinPct = ("Win", "mean"),
                  ASTpg = ("Ast", "mean"),
                  TOpg = ("TO", "mean"),
                  STLpg = ("Stl", "mean"),
                  BLKpg = ("Blk", "mean"),).reset_index()
 
    # Shooting %s need sums before dividing
    shoot = grp[["FGM","FGA","FGM3","FGA3"]].sum().reset_index()
    shoot["FGpct"]  = shoot["FGM"]  / shoot["FGA"]
    shoot["FG3pct"] = shoot["FGM3"] / shoot["FGA3"]
    agg = agg.merge(shoot[["Season","TeamID","FGpct","FG3pct"]], on = ["Season","TeamID"])
 
    agg["PointDiff"] = agg["PointsFor"] - agg["PointsAgn"]
 
    # Four Factors
    ff_rows = []
    for (season, team), g in tall.groupby(["Season", "TeamID"]):
        row = _four_factors(g, tall)
        row["Season"] = season
        row["TeamID"] = team
        ff_rows.append(row)
    ff = pd.DataFrame(ff_rows)
 
    agg = agg.merge(ff, on=["Season","TeamID"], how="left")
    return agg

# now functions to including ranking systems
# setting ranking systems to use
# Pomeroy, Sag-Elo, Massey, Markov Chain, Moore, Wolfe
MASSEY_SYSTEMS = ["POM", "MAS", "MOR", "MLK"]
EXCLUDE_FROM_DIFFS = {"MadeTourney", "rank_composite"}

# function to make ranking based on other ranking systems
def build_massey_features(gender = "M"):
    """
    Extract end-of-season Massey ordinal rankings for selected systems.
    """
    # Only men's has MMasseyOrdinals; women's does not
    if gender == "W":
        return pd.DataFrame(columns = ["Season", "TeamID"])
 
    massey = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
 
    # pre-tournament rankings of systems previously defined 
    massey = massey[(massey["SystemName"].isin(MASSEY_SYSTEMS)) & (massey["RankingDayNum"] <= 133)]
 
    # For each team/season/system, take the latest available ranking
    massey = (massey.sort_values("RankingDayNum").groupby(["Season", "TeamID", "SystemName"]).last().reset_index())
 
    # Pivot so each system becomes its own column
    pivoted = massey.pivot_table(index = ["Season", "TeamID"],
                                 columns = "SystemName",
                                 values = "OrdinalRank").reset_index()
    pivoted.columns.name = None
    pivoted.columns = (["Season", "TeamID"] + [f"rank_{s}" for s in pivoted.columns[2:]])
 
    # making composite ranking by taking mean across systems
    rank_cols = [c for c in pivoted.columns if c.startswith("rank_")]
    pivoted["rank_composite"] = pivoted[rank_cols].mean(axis = 1)
 
    return pivoted


# functions to extract tournament seeds from teams
def _parse_seed(s):
    # removes the region aspect of seeds
    return int("".join(filter(str.isdigit, str(s))))

# main function to get seeds for teams based on year/season
def build_seed_features(gender = "M"):
    seeds = pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")
    seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed)
    return seeds[["Season", "TeamID", "SeedNum"]]

# MAIN DATA FUNCTION 
# Combines all functions together to get finalized of the appropriate stat features
# MAIN DATA FUNCTION 
def build_team_features(gender = "M", save = True):
    ##
    # remove print statements when ensure that everything works
    ##
    print(f"[{gender}] Building box-score features...")
    box = build_box_features(gender)
 
    print(f"[{gender}] Building Massey ordinal features...")
    massey = build_massey_features(gender)
 
    print(f"[{gender}] Building seed features...")
    seeds = build_seed_features(gender)
 
    # Merge — seeds only exist for tournament teams, so left-join on box
    df = box.merge(massey, on = ["Season", "TeamID"], how = "left")
    df = df.merge(seeds, on = ["Season", "TeamID"], how = "left")
 
    # Teams that didn't make the tournament get seed = 99 (clearly worse)
    df["SeedNum"] = df["SeedNum"].fillna(99).astype(int)
 
    # new boolean to separate between tournament and non-tournament teams
    df["MadeTourney"] = (df["SeedNum"] < 99).astype(int)
 
    # save new file to processed/ directory
    if save:
        out = PROC_DIR / f"{gender}_team_features.csv"
        df.to_csv(out, index = False)
        print(f"[{gender}] Saved {out}, shape = {df.shape}")
 
    return df

# now, all functions below relating to setting up data properly for submission
# getting id's from submission files, setting up match-ups based on id

# function to get team_ids, from submission file:
def _parse_matchup_id(mid):
    """'2026_1234_5678' → (2026, 1234, 5678)"""
    parts = mid.split("_")
    # into year, team1, team2
    return int(parts[0]), int(parts[1]), int(parts[2])

def build_matchup_df(team_features, gender = "M", mode = "train"):
    """
    Build a matchup-level DataFrame where each row is a game and
    features are the *difference* (TeamA - TeamB) in season stats.
    """
    prefix = gender
 
    # get all columns except the year/team id for stats
    feat_cols = [c for c in team_features.columns if c not in ["Season", "TeamID"] and c not in EXCLUDE_FROM_DIFFS]
 
    if mode == "train":
        tourney = pd.read_csv(DATA_DIR / f"{prefix}NCAATourneyCompactResults.csv")

        # keeping data restricted to as much data is entirely covered (2003-current)
        valid_seasons = sorted(team_features["Season"].unique())
        tourney = tourney[tourney["Season"].isin(valid_seasons)]

        # Assign TeamA = lower ID, TeamB = higher ID
        # matches sample submission convention
        # Label for if TeamA was winning team
        tourney["TeamA"] = tourney[["WTeamID", "LTeamID"]].min(axis = 1)
        tourney["TeamB"] = tourney[["WTeamID", "LTeamID"]].max(axis = 1)
        tourney["Label"] = (tourney["WTeamID"] == tourney["TeamA"]).astype(int)
        source = tourney[["Season", "TeamA", "TeamB", "Label"]]
    else:
        # for testing, use submissions csv
        sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
        parsed = sub["ID"].apply(_parse_matchup_id)
        source = pd.DataFrame(parsed.tolist(), columns = ["Season", "TeamA", "TeamB"])
        source["ID"] = sub["ID"].values
        source["Label"] = np.nan
 
        # Filter: men's TeamIDs are 1000-1999, women's are 3000-3999
        if gender == "M":
            source = source[source["TeamA"] < 2000].reset_index(drop=True)
        else:
            source = source[source["TeamA"] >= 3000].reset_index(drop=True)
 
    # merge features for team a
    tf_a = team_features.rename(columns = {"TeamID": "TeamA", **{c: f"A_{c}" for c in feat_cols}})
    tf_b = team_features.rename(columns = {"TeamID": "TeamB", **{c: f"B_{c}" for c in feat_cols}})

    keep_a = ["Season", "TeamA"] + [f"A_{c}" for c in feat_cols]
    keep_b = ["Season", "TeamB"] + [f"B_{c}" for c in feat_cols]
    tf_a = tf_a[[c for c in keep_a if c in tf_a.columns]]
    tf_b = tf_b[[c for c in keep_b if c in tf_b.columns]]
 
    df = source.merge(tf_a, on = ["Season", "TeamA"], how = "left")
    df = df.merge(tf_b, on = ["Season", "TeamB"], how = "left")
 
    # convert individual team features into difference of two groups
    # features are now (A - B)
    diff_cols = []
    for col in feat_cols:
        a_col = f"A_{col}"
        b_col = f"B_{col}"
        if a_col in df.columns and b_col in df.columns:
            diff_name = f"diff_{col}"
            df[diff_name] = df[a_col] - df[b_col]
            diff_cols.append(diff_name)
 
    print(f"[{gender}] Matchup df shape: {df.shape}  |  diff features: {len(diff_cols)}")
 
    return df, diff_cols

# finally, join all dfs into one for both genders and save
def build_all(genders = ("M", "W")):
    """Build and save team features for both genders."""
    results = {}
    for g in genders:
        results[g] = build_team_features(gender = g, save = True)
    return results
 
# run all above functionswhen run
if __name__ == "__main__":
    build_all()