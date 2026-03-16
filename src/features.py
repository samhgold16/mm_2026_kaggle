# important packages
import pandas as pd
import numpy as np
from pathlib import Path
import re

# setting pathing
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# setting years for analysis
# TRAIN_SEASONS = list(range(2003, 2026))
# PRED_SEASON   = 2026 

# setting ranking systems to use
# Pomeroy, Sag-Elo, Massey, Markov Chain, Moore, Wolfe
MASSEY_SYSTEMS = ["POM", "SAG", "MAS", "MOR", "RPI"]
EXCLUDE_FROM_DIFFS = {"MadeTourney", "rank_composite", "TeamID", "Season"}
RECENT_GAMES_N = 10

# helper
def _estimate_possessions(fga, fta, orb, to, opp_drb):
    return fga - orb + to + 0.44 * fta

# functions to process data and extract info
def _stack_games(df):
    """
    Convert winner/loser wide format into two rows per game (one per team).
    """
    # choosing major stats to consider
    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
 
    # extracting major stats for winning team
    winner_cols = (["Season", "DayNum", "WTeamID", "WScore", "LScore"] +
                   [f"W{c}" for c in stat_cols] +
                   ["LOR", "LDR", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LTO"])
    winner = df[winner_cols].copy()
    winner.columns = (["Season", "DayNum", "TeamID", "Score", "OppScore"] +
                      stat_cols +
                      ["OppOR", "OppDR", "OppFGM", "OppFGA", "OppFGM3", "OppFGA3",
                       "OppFTM", "OppFTA", "OppTO"])
    # setting binary win/loss column
    winner["Win"] = 1
 
    # extracting major stats for losing team
    loser_cols = (["Season", "DayNum", "LTeamID", "LScore", "WScore"] +
                  [f"L{c}" for c in stat_cols] +
                  ["WOR", "WDR", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WTO"])

    loser = df[loser_cols].copy()
    loser.columns = (["Season", "DayNum", "TeamID", "Score", "OppScore"] +
                     stat_cols +
                     ["OppOR", "OppDR", "OppFGM", "OppFGA", "OppFGM3", "OppFGA3",
                      "OppFTM", "OppFTA", "OppTO"])
    # setting binary win/loss column
    loser["Win"] = 0
 
    return pd.concat([winner, loser], ignore_index = True)

# function to extract more unique factors
# offense
def _compute_four_factors_offensive(group_df):
    """
    dean oliver's four unique factors
    """
    fga = group_df["FGA"].sum()
    fgm = group_df["FGM"].sum()
    fgm3 = group_df["FGM3"].sum()
    ftm = group_df["FTM"].sum()
    fta = group_df["FTA"].sum()
    to = group_df["TO"].sum()
    orb = group_df["OR"].sum()
    opp_drb = group_df["OppDR"].sum()

    efg = (fgm + 0.5 * fgm3) / fga if fga > 0 else np.nan
    tov = to / (fga + 0.44 * fta + to) if (fga + 0.44 * fta + to) > 0 else np.nan
    orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) > 0 else np.nan
    ftr = ftm / fga if fga > 0 else np.nan

    return pd.Series({"eFG_pct": efg,
                      "TOV_pct": tov,
                      "ORB_pct": orb_pct,
                      "FTR": ftr})

# defense
def _compute_four_factors_defensive(group_df):
    """
    same as above for defensive stats
    """
    opp_fga = group_df["OppFGA"].sum()
    opp_fgm = group_df["OppFGM"].sum()
    opp_fgm3 = group_df["OppFGM3"].sum()
    opp_ftm = group_df["OppFTM"].sum()
    opp_fta = group_df["OppFTA"].sum()
    opp_to = group_df["OppTO"].sum()
    drb = group_df["DR"].sum()
    opp_orb = group_df["OppOR"].sum()

    opp_efg = (opp_fgm + 0.5 * opp_fgm3) / opp_fga if opp_fga > 0 else np.nan
    opp_tov = opp_to / (opp_fga + 0.44 * opp_fta + opp_to) if (opp_fga + 0.44 * opp_fta + opp_to) > 0 else np.nan
    drb_pct = drb / (drb + opp_orb) if (drb + opp_orb) > 0 else np.nan
    opp_ftr = opp_ftm / opp_fga if opp_fga > 0 else np.nan

    return pd.Series({"Opp_eFG_pct": opp_efg,
                     "Opp_TOV_pct": opp_tov,  
                     "DRB_pct": drb_pct,
                     "Opp_FTR": opp_ftr})

# more metrics, effiency stuff
def _compute_efficiency(group_df):
    """
    Compute tempo and efficiency metrics
    """
    n_games = len(group_df)

    # Estimate possessions for each game
    fga = group_df["FGA"].sum()
    fta = group_df["FTA"].sum()
    orb = group_df["OR"].sum()
    to = group_df["TO"].sum()
    opp_drb = group_df["OppDR"].sum()

    total_poss = _estimate_possessions(fga, fta, orb, to, opp_drb)

    # Also compute opponent possessions for symmetry
    opp_fga = group_df["OppFGA"].sum()
    opp_fta = group_df["OppFTA"].sum()
    opp_orb = group_df["OppOR"].sum()
    opp_to = group_df["OppTO"].sum()
    drb = group_df["DR"].sum()

    opp_poss = _estimate_possessions(opp_fga, opp_fta, opp_orb, opp_to, drb)

    # Average possessions (more stable)
    avg_poss = (total_poss + opp_poss) / 2

    # Points
    pts_for = group_df["Score"].sum()
    pts_against = group_df["OppScore"].sum()

    # Tempo = possessions per game
    tempo = avg_poss / n_games if n_games > 0 else np.nan

    # Efficiency = points per 100 possessions
    off_eff = (pts_for / avg_poss) * 100 if avg_poss > 0 else np.nan
    def_eff = (pts_against / avg_poss) * 100 if avg_poss > 0 else np.nan

    return pd.Series({"Tempo": tempo,
                      "OffEff": off_eff,
                      "DefEff": def_eff,
                      "NetEff": off_eff - def_eff if (off_eff and def_eff) else np.nan})

# Main Feature Building Functions

# main function to combine above internal functions
def build_box_features(gender = "M"):
    """
    Compute per-team, per-season stats from REGULAR SEASON games only.
    """

    path = DATA_DIR / f"{gender}RegularSeasonDetailedResults.csv"
    reg = pd.read_csv(path)
    tall = _stack_games(reg)
 
    grp = tall.groupby(["Season", "TeamID"])
 
    agg = grp.agg(Games = ("Win", "count"),
                  Wins=("Win", "sum"),
                  PointsFor=("Score", "mean"),
                  PointsAgn=("OppScore", "mean"),
                  WinPct=("Win", "mean"),
                  ASTpg=("Ast", "mean"),
                  TOpg=("TO", "mean"),
                  STLpg=("Stl", "mean"),
                  BLKpg=("Blk", "mean"),
                  ORpg=("OR", "mean"),
                  DRpg=("DR", "mean"),
                  PFpg=("PF", "mean")).reset_index()
    
    agg["PointDiff"] = agg["PointsFor"] - agg["PointsAgn"]
 
    # Shooting %s need sums before dividing
    shoot = grp[["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA"]].sum().reset_index()
    shoot["FGpct"] = shoot["FGM"] / shoot["FGA"]
    shoot["FG3pct"] = shoot["FGM3"] / shoot["FGA3"]
    shoot["FTpct"] = shoot["FTM"] / shoot["FTA"]
    shoot["FG3Rate"] = shoot["FGA3"] / shoot["FGA"]  # 3-point attempt rate
    agg = agg.merge(shoot[["Season", "TeamID", "FGpct", "FG3pct", "FTpct", "FG3Rate"]],
                    on=["Season", "TeamID"])
   # Four Factors - Offensive
    print(f"[{gender}] Computing offensive Four Factors...")
    off_ff = grp.apply(_compute_four_factors_offensive).reset_index()
    agg = agg.merge(off_ff, on=["Season", "TeamID"], how="left")

    # Four Factors - Defensive
    print(f"[{gender}] Computing defensive Four Factors...")
    def_ff = grp.apply(_compute_four_factors_defensive).reset_index()
    agg = agg.merge(def_ff, on=["Season", "TeamID"], how="left")

    # Tempo and Efficiency
    print(f"[{gender}] Computing tempo and efficiency...")
    eff = grp.apply(_compute_efficiency).reset_index()
    agg = agg.merge(eff, on=["Season", "TeamID"], how="left")

    return agg, tall

def build_recent_form(tall, n_games=RECENT_GAMES_N):
    """
    Compute performance over the last N games of the regular season.
    """
    # Sort by date and get last N games per team-season
    tall_sorted = tall.sort_values(["Season", "TeamID", "DayNum"])
    recent = tall_sorted.groupby(["Season", "TeamID"]).tail(n_games)

    # Aggregate recent performance
    grp = recent.groupby(["Season", "TeamID"])

    form = grp.agg(Recent_WinPct=("Win", "mean"),
                  Recent_PointDiff=("Score", lambda x: (x - recent.loc[x.index, "OppScore"]).mean()),
                  Recent_Games=("Win", "count")).reset_index()

    # Add recent offensive/defensive efficiency
    recent_eff = grp.apply(_compute_efficiency).reset_index()
    recent_eff.columns = ["Season", "TeamID", "Recent_Tempo", "Recent_OffEff",
                          "Recent_DefEff", "Recent_NetEff"]

    form = form.merge(recent_eff, on=["Season", "TeamID"], how="left")

    # Momentum = Recent performance vs season average (computed later via merge)
    return form

# now functions to including ranking systems
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

    # Also compute rank std (disagreement between systems can be informative)
    pivoted["rank_std"] = pivoted[rank_cols].std(axis = 1)
 
    return pivoted

# main function to get seeds for teams based on year/season
def build_seed_features(gender="M"):
    """
    Parse tournament seeds and compute seed-based features.
    """
    print(f"[{gender}] Loading seed data...")
    seeds = pd.read_csv(DATA_DIR / f"{gender}NCAATourneySeeds.csv")
    seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed)

    # Extract region (first character)
    seeds["Region"] = seeds["Seed"].str[0]

    return seeds[["Season", "TeamID", "SeedNum", "Region"]]

# functions to extract tournament seeds from teams
def _parse_seed(s):
    # removes the region aspect of seeds
    s = str(s).strip()
    # Remove leading region letter (single uppercase letter at start)
    s = re.sub(r'^[A-Z]', '', s)
    # Remove trailing play-in letters (a, b)
    s = re.sub(r'[ab]$', '', s, flags=re.IGNORECASE)
    return int(s)

# MAIN DATA FUNCTION 
# Combines all functions together to get finalized of the appropriate stat features
# MAIN DATA FUNCTION 
def build_team_features(gender = "M", save = True):
    ##
    # remove print statements when ensure that everything works
    ##
    # Box score features (including Four Factors and efficiency)
    box, tall = build_box_features(gender)

    # Recent form
    form = build_recent_form(tall)

    # Massey ordinals
    massey = build_massey_features(gender)

    # Seeds
    seeds = build_seed_features(gender)

    # --- Merge all features ---
    print(f"[{gender}] Merging all features...")
    df = box.merge(form, on=["Season", "TeamID"], how="left")
    df = df.merge(massey, on=["Season", "TeamID"], how="left")
    df = df.merge(seeds, on=["Season", "TeamID"], how="left")

    # Teams that didn't make tournament get seed = 99
    df["SeedNum"] = df["SeedNum"].fillna(99).astype(int)
    df["MadeTourney"] = (df["SeedNum"] < 99).astype(int)

    # Compute momentum (recent form vs season average)
    df["Momentum_WinPct"] = df["Recent_WinPct"] - df["WinPct"]
    df["Momentum_NetEff"] = df["Recent_NetEff"] - df["NetEff"]

    # Save
    if save:
        out = PROC_DIR / f"{gender}_team_features.csv"
        df.to_csv(out, index=False)
        print(f"[{gender}] Saved to {out}")
        print(f"[{gender}] Shape: {df.shape}")
        print(f"[{gender}] Columns: {list(df.columns)}")

    return df

# now, all functions below relating to setting up data properly for submission
# getting id's from submission files, setting up match-ups based on id

# function to get team_ids, from submission file:
def _parse_matchup_id(mid):
    """'2026_1234_5678' → (2026, 1234, 5678)"""
    parts = mid.split("_")
    # into year, team1, team2
    return int(parts[0]), int(parts[1]), int(parts[2])

def build_matchup_df(team_features, gender="M", mode="train",
                     include_interactions=True):
    """
    Build matchup-level DataFrame with:
    - Difference features (TeamA - TeamB)
    - Ratio features (for scale-invariant comparisons)
    - Interaction features (style matchups)
    - Historical seed upset prior
    """
    print(f"\n[{gender}] Building matchup DataFrame (mode={mode})...")

    # Get feature columns (exclude non-numeric and special columns)
    feat_cols = [c for c in team_features.columns
                 if c not in EXCLUDE_FROM_DIFFS
                 and c not in ["Region"]
                 and team_features[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Load source data
    if mode == "train":
        tourney = pd.read_csv(DATA_DIR / f"{gender}NCAATourneyCompactResults.csv")
        valid_seasons = sorted(team_features["Season"].unique())
        tourney = tourney[tourney["Season"].isin(valid_seasons)]

        # TeamA = lower ID (matches Kaggle submission format)
        tourney["TeamA"] = tourney[["WTeamID", "LTeamID"]].min(axis=1)
        tourney["TeamB"] = tourney[["WTeamID", "LTeamID"]].max(axis=1)
        tourney["Label"] = (tourney["WTeamID"] == tourney["TeamA"]).astype(int)
        source = tourney[["Season", "TeamA", "TeamB", "Label"]]
    else:
        sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
        parsed = sub["ID"].apply(_parse_matchup_id)
        source = pd.DataFrame(parsed.tolist(), columns=["Season", "TeamA", "TeamB"])
        source["ID"] = sub["ID"].values
        source["Label"] = np.nan

        # Filter by gender (Men: <2000, Women: >=3000)
        if gender == "M":
            source = source[source["TeamA"] < 2000].reset_index(drop=True)
        else:
            source = source[source["TeamA"] >= 3000].reset_index(drop=True)

    # --- Merge team features ---
    tf_a = team_features.rename(
        columns={"TeamID": "TeamA", **{c: f"A_{c}" for c in feat_cols}}
    )
    tf_b = team_features.rename(
        columns={"TeamID": "TeamB", **{c: f"B_{c}" for c in feat_cols}}
    )

    keep_a = ["Season", "TeamA"] + [f"A_{c}" for c in feat_cols]
    keep_b = ["Season", "TeamB"] + [f"B_{c}" for c in feat_cols]
    tf_a = tf_a[[c for c in keep_a if c in tf_a.columns]]
    tf_b = tf_b[[c for c in keep_b if c in tf_b.columns]]

    df = source.merge(tf_a, on=["Season", "TeamA"], how="left")
    df = df.merge(tf_b, on=["Season", "TeamB"], how="left")

    # --- Compute difference features ---
    diff_data = {}
    diff_cols = []
    for col in feat_cols:
        a_col, b_col = f"A_{col}", f"B_{col}"
        if a_col in df.columns and b_col in df.columns:
            diff_name = f"diff_{col}"
            diff_data[diff_name] = df[a_col] - df[b_col]
            diff_cols.append(diff_name)

    # --- Compute ratio features (for key metrics) ---
    ratio_metrics = ["OffEff", "DefEff", "Tempo", "eFG_pct", "WinPct"]
    ratio_data = {}
    ratio_cols = []
    for col in ratio_metrics:
        a_col, b_col = f"A_{col}", f"B_{col}"
        if a_col in df.columns and b_col in df.columns:
            ratio_name = f"ratio_{col}"
            ratio_data[ratio_name] = df[a_col] / df[b_col].replace(0, np.nan)
            ratio_cols.append(ratio_name)

    # --- Compute interaction features (style matchups) ---
    interaction_data = {}
    interaction_cols = []
    if include_interactions:
        if "diff_Tempo" in diff_data and "diff_NetEff" in diff_data:
            interaction_data["interact_tempo_eff"] = diff_data["diff_Tempo"] * diff_data["diff_NetEff"]
            interaction_cols.append("interact_tempo_eff")

        if "diff_ORB_pct" in diff_data and "diff_Opp_FTR" in diff_data:
            interaction_data["interact_physical"] = diff_data["diff_ORB_pct"] * diff_data["diff_Opp_FTR"]
            interaction_cols.append("interact_physical")

        if "A_SeedNum" in df.columns and "B_SeedNum" in df.columns:
            interaction_data["SeedDiff"] = df["A_SeedNum"] - df["B_SeedNum"]
            interaction_cols.append("SeedDiff")
        
    # --- Concat all new columns at once ---
    new_cols = {**diff_data, **ratio_data, **interaction_data}
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Defragment after the merge
    df = df.copy()

    all_feature_cols = diff_cols + ratio_cols + interaction_cols

    print(f"[{gender}] Matchup shape: {df.shape}")
    print(f"[{gender}] Feature breakdown:")
    print(f"         Diff features: {len(diff_cols)}")
    print(f"         Ratio features: {len(ratio_cols)}")
    print(f"         Interaction features: {len(interaction_cols)}")
    print(f"         Total: {len(all_feature_cols)}")

    return df, all_feature_cols

def build_all(genders=("M", "W")):
    """Build and save team features for specified genders."""
    results = {}
    for g in genders:
        results[g] = build_team_features(gender=g, save=True)
    return results
 
# run all above functionswhen run
if __name__ == "__main__":
    build_all()