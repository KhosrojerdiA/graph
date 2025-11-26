

import sys
import os
main_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(main_path)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#______________________________________________________________________________________________________________________



def prepare_df(df, data_name_value, graph_model_value, eps=1e-9):
    """
    Prepare a comparison DataFrame by:
      - filtering for a specific dataset (data_name) and graph model (graph_model),
      - sorting each model_name's 'delta' column independently (descending),
      - removing zero or near-zero delta values.
    
    Returns a DataFrame with columns: Query, Degree, Page Rank, Viking, Proposed Method
    """
    # Filter for dataset and graph model
    temp = df[(df["data_name"] == data_name_value) & 
              (df["graph_model"] == graph_model_value)].copy()
    
    # Ensure delta is numeric
    temp["delta"] = pd.to_numeric(temp["delta"], errors="coerce")
    
    sorted_cols = {}
    max_len = 0
    model_order = ["Degree", "Page Rank", "Viking", "Proposed Method"]

    # Sort and clean each model group
    for model in model_order:
        sub = temp[temp["model_name"] == model]["delta"]
        # Remove near-zero and NaN values, then sort descending
        clean_values = sub[(sub.abs() > eps) & (~sub.isna())].sort_values(ascending=False).to_numpy()
        sorted_cols[model] = clean_values
        max_len = max(max_len, len(clean_values))

    # Construct output DataFrame
    out = pd.DataFrame({"Query": range(1, max_len + 1)})
    for model in model_order:
        padded = list(sorted_cols.get(model, [])) + [np.nan] * (max_len - len(sorted_cols.get(model, [])))
        out[model] = padded

    return out

#______________________________________________________________________________________________________________________

result_path = f"{main_path}/outputs/run_v1"

# --- Save best-quality versions ---
save_dir = f"{main_path}/Visualization"

# Read the csv file
df = pd.read_csv(f'{result_path}/query_positions/all_query_positions_combined.csv')
df.head()

#______________________________________________________________________________________________________________________

#data_name,model_name,graph_model,node_id,retrieval_position,attack_position,delta,success_flag

''' 
Cora_GCN = df[(df["data_name"] == "Cora") & (df["graph_model"] == "GCN")]
Cora_GCN = Cora_GCN.loc[:, ~Cora_GCN.columns.isin(["data_name", "graph_model"])]

Cora_GraphSage = df[(df["data_name"] == "Cora") & (df["graph_model"] == "GraphSage")]
Cora_GraphSage = Cora_GraphSage.loc[:, ~Cora_GraphSage.columns.isin(["data_name", "graph_model"])]

CiteSeer_GCN = df[(df["data_name"] == "CiteSeer") & (df["graph_model"] == "GCN")]
CiteSeer_GCN = CiteSeer_GCN.loc[:, ~CiteSeer_GCN.columns.isin(["data_name", "graph_model"])]

CiteSeer_GraphSage = df[(df["data_name"] == "CiteSeer") & (df["graph_model"] == "GraphSage")]
CiteSeer_GraphSage = CiteSeer_GraphSage.loc[:, ~CiteSeer_GraphSage.columns.isin(["data_name", "graph_model"])]

PubMed_GCN = df[(df["data_name"] == "PubMed") & (df["graph_model"] == "GCN")]
PubMed_GCN = PubMed_GCN.loc[:, ~PubMed_GCN.columns.isin(["data_name", "graph_model"])]

PubMed_GraphSage = df[(df["data_name"] == "PubMed") & (df["graph_model"] == "GraphSage")]
PubMed_GraphSage = PubMed_GraphSage.loc[:, ~PubMed_GraphSage.columns.isin(["data_name", "graph_model"])]
'''

Cora_GCN = prepare_df(df, "Cora", "GCN")
Cora_GraphSage = prepare_df(df, "Cora", "GraphSage")
CiteSeer_GCN = prepare_df(df, "CiteSeer", "GCN")
CiteSeer_GraphSage = prepare_df(df, "CiteSeer", "GraphSage")
PubMed_GCN = prepare_df(df, "PubMed", "GCN")
PubMed_GraphSage = prepare_df(df, "PubMed", "GraphSage")


print(Cora_GCN)
print(Cora_GraphSage)
print(CiteSeer_GCN)
print(CiteSeer_GraphSage)
print(PubMed_GCN)
print(PubMed_GraphSage)

#______________________________________________________________________________________________________________________

# Arrange GCNs in first row, GraphSAGEs in second
dfs = [
    (Cora_GCN, "Cora - GCN"),
    (CiteSeer_GCN, "CiteSeer - GCN"),
    (PubMed_GCN, "PubMed - GCN"),
    (Cora_GraphSage, "Cora - GraphSAGE"),
    (CiteSeer_GraphSage, "CiteSeer - GraphSAGE"),
    (PubMed_GraphSage, "PubMed - GraphSAGE")
]

colors = {
    "Degree": "red",
    "Page Rank": "blue",
    "Viking": "green"
}

linestyles = {
    "Degree": "-",
    "Page Rank": "--",
    "Viking": "-."
}

# --- Font and style scaling ---
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 3.6,
})

# --- Figure setup ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
handles, labels = None, None

for i, (df, title) in enumerate(dfs):
    ax = axes.flat[i]

    for col in ["Degree", "Page Rank", "Viking"]:
        line, = ax.plot(
            df["Query"], df[col],
            label=col,
            color=colors[col],
            linestyle=linestyles[col],
            linewidth=3.6
        )
        if i == 0:  # capture handles once for legend
            handles = handles or []
            labels = labels or []
            handles.append(line)
            labels.append(f"vs {col}")

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel("Query", fontsize=20)
    ax.set_ylabel(r"$\Delta$", fontsize=28, rotation=0, labelpad=22)
    ax.grid(True, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks([])
    ax.tick_params(axis='x', length=0)



    y_min = df[["Degree", "Page Rank", "Viking"]].min().min()
    y_max = df[["Degree", "Page Rank", "Viking"]].max().max()

    # Expand lower bound if all values are positive
    if y_min >= 0:
        y_min = -0.1 * y_max
        ax.set_ylim(y_min, y_max)

    # Ensure negative ticks are visible and properly formatted
    yticks = ax.get_yticks()
    
    # For PubMed plots, always ensure -50 is visible and remove -100
    if "PubMed" in title:
        # Remove -100 if present
        yticks = yticks[yticks != -100]
        # Add -50 if not present
        #if -50 not in yticks:
            #yticks = np.append(yticks, -50)
        yticks = np.sort(yticks)
        ax.set_yticks(yticks)
    # For other plots, add negative ticks if missing
    elif not np.any(yticks < 0):
        # Add a negative tick at a reasonable position
        if y_min < 0:
            neg_tick = round(y_min * 0.5, 2)  # halfway toward lower limit
        else:
            neg_tick = round(-0.1 * y_max, 2)  # 10% below zero
        
        yticks = np.append(yticks, neg_tick)
        yticks = np.sort(yticks)
        ax.set_yticks(yticks)
    
    # Ensure y-axis ticks are clearly visible
    ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)
    ax.tick_params(axis='y', which='minor', length=3, width=1)

    # Visually highlight 0-line
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.9)


# --- Layout & Legend ---
fig.tight_layout(rect=[0, 0.12, 1, 0.95])
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    frameon=False,
    fontsize=22,
    bbox_to_anchor=(0.5, 0.05)
)
#______________________________________________________________________________________________________________________


pdf_path = f"{save_dir}/performance_comparison.pdf"
png_path = f"{save_dir}/performance_comparison.png"

fig.savefig(
    pdf_path,
    format="pdf",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True
)
fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02
)

print(f"✅ Vector (PDF) and PNG saved:\n- {pdf_path}\n- {png_path}")
plt.show()
#______________________________________________________________________________________________________________________