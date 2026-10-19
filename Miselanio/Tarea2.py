import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# Configuración
plt.style.use("default")
plt.rcParams.update({
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 1.2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,
    "legend.frameon": True
})
df = pd.read_csv("industry_dataset_structured.csv")


line_colors = {"Line_A":"#1f77b4", "Line_B":"#ff7f0e", "Line_C":"#2ca02c"}
line_markers = {"Line_A":"o", "Line_B":"s", "Line_C":"^"}


shift_styles = {
    "Morning": {"color":"#9467bd", "linestyle":"-"},
    "Evening": {"color":"#8c564b", "linestyle":"--"},
    "Night": {"color":"#d62728", "linestyle":"-."}
}




# Figura 1
fig, ax = plt.subplots()
for line, data in df.groupby("Production_Line"):
    ax.scatter(data["Units_Produced"], data["Defect_Rate"],
               c=line_colors[line], marker=line_markers[line],
               edgecolor="black", alpha=0.7, label=line)


ax.set_title("Fig. 1: Units_Produced vs Defect_Rate")
ax.set_xlabel("Units_Produced")
ax.set_ylabel("Defect_Rate")
ax.legend(title="Production_Line")
ax.grid(True)
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
plt.show()




# Figura 2
fig, ax = plt.subplots()
ax.scatter(df["Energy_Consumption_kWh"], df["Revenue_USD"],
           facecolor="lightgreen", edgecolor="black", label="Observaciones")


x, y = df["Energy_Consumption_kWh"], df["Revenue_USD"]
m, b = np.polyfit(x, y, 1)
ax.plot(x, m*x+b, "--", color="darkgreen", label="Tendencia (OLS)")
r = np.corrcoef(x, y)[0,1]


ax.set_title(f"Fig. 2: Energía vs Ingresos (r = {r:.2f})")
ax.set_xlabel("Energy_Consumption_kWh")
ax.set_ylabel("Revenue_USD")
ax.get_yaxis().set_major_formatter(lambda v, _: f"{int(v):,}")
ax.legend()
ax.grid(True)
plt.show()




# Figura 3
fig, ax = plt.subplots()
sns.histplot(df["Defect_Rate"], bins=20, color="lightblue", edgecolor="darkblue", ax=ax)
mean, median = df["Defect_Rate"].mean(), df["Defect_Rate"].median()
ax.axvline(mean, color="green", linestyle="--", label=f"Media {mean:.2%}")
ax.axvline(median, color="red", linestyle="-.", label=f"Mediana {median:.2%}")
ax.set_title("Fig. 3: Distribución de Defect_Rate")
ax.set_xlabel("Defect_Rate")
ax.set_ylabel("Frecuencia")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.legend()
ax.grid(True)
plt.show()




# Figura 4
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(df["Downtime_Minutes"], bins=15,
                                color="#f28e2b", edgecolor="#1a1a1a", linewidth=1.2)
mode_bin = bins[np.argmax(counts)]
p90, p95 = np.percentile(df["Downtime_Minutes"], [90,95])
ax.axvline(mode_bin, color="green", linestyle="--", linewidth=1.5, label=f"Moda ({mode_bin:.1f})")
ax.axvline(p90, color="blue", linestyle=":", linewidth=1.3, label=f"P90 ({p90:.1f})")
ax.axvline(p95, color="red", linestyle="-.", linewidth=1.3, label=f"P95 ({p95:.1f})")
ax.fill_betweenx([0, max(counts)], p95, max(df["Downtime_Minutes"]),
                 color="red", alpha=0.3, label="Cola derecha")
sns.kdeplot(df["Downtime_Minutes"], ax=ax, color="#111", linewidth=2, label="Densidad (KDE)")
ax.set_title("Fig. 4: Distribución de Downtime_Minutes")
ax.set_xlabel("Downtime_Minutes")
ax.set_ylabel("Frecuencia")
ax.grid(True)
ax.legend()
plt.show()




# Figura 5
fig, ax = plt.subplots()
group = df.groupby("Production_Line")["Units_Produced"]
means = group.mean()
errors = group.sem()  # EE
means.sort_values().plot.bar(yerr=errors[means.index], ax=ax,
                             color=[line_colors[i] for i in means.index], edgecolor="black")
for i, v in enumerate(means.sort_values()):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
ax.set_title("Fig. 5: Productividad promedio por línea (±EE)")
ax.set_ylabel("Units_Produced (promedio)")
ax.grid(True)
plt.show()




# Figura 6
fig, ax = plt.subplots()
group = df.groupby("Shift")["Revenue_USD"]
means = group.mean()
errors = group.sem()
means.plot.bar(yerr=errors, ax=ax, color=[shift_styles[s]["color"] for s in means.index], edgecolor="black")
for i, v in enumerate(means):
    ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
ax.set_title("Fig. 6: Ingresos promedio por turno (±EE)")
ax.set_ylabel("Revenue_USD (promedio)")
ax.grid(True)
plt.show()




# Figura 7
fig, ax = plt.subplots()
sns.boxplot(x="Shift", y="Cost_per_Unit", data=df, notch=True, ax=ax,
            palette={s:shift_styles[s]["color"] for s in shift_styles})
ax.set_title("Fig. 7: Cost_per_Unit por Shift (Boxplot)")
ax.grid(True)
plt.show()




# Figura 8
fig, ax = plt.subplots()
sns.boxplot(x="Production_Line", y="Defect_Rate", hue="Shift", data=df,
            palette={s:shift_styles[s]["color"] for s in shift_styles})
ax.set_title("Fig. 8: Defect_Rate por línea y turno (Boxplots)")
ax.set_ylabel("Defect_Rate")
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.grid(True)
plt.legend(title="Shift")
plt.show()




# Figura 9
fig, ax = plt.subplots()
sns.kdeplot(df["Energy_Consumption_kWh"], ax=ax, fill=True, alpha=0.3)
mean, median = df["Energy_Consumption_kWh"].mean(), df["Energy_Consumption_kWh"].median()
ax.axvline(mean, linestyle="--", color="green", label=f"Media {mean:.1f}")
ax.axvline(median, linestyle="-.", color="red", label=f"Mediana {median:.1f}")
ax.set_title("Fig. 9: Densidad de Energy_Consumption_kWh")
ax.set_ylabel("Densidad")
ax.legend()
ax.grid(True)
plt.show()




# Figura 10
fig, ax = plt.subplots()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label":"Correlation (r)"})
ax.set_title("Fig. 10: Heatmap de correlación")
plt.show()




# Figura 11
fig, ax = plt.subplots()
for line, data in df.groupby("Production_Line"):
    grouped = data.groupby("Batch")["Units_Produced"]
    mean = grouped.mean()
    err = grouped.sem()
    ax.plot(mean.index, mean, label=line, color=line_colors[line], marker=line_markers[line])
    ax.fill_between(mean.index, mean-err, mean+err, color=line_colors[line], alpha=0.2)
ax.set_title("Fig. 11: Tendencias Units_Produced por Línea (±EE)")
ax.set_xlabel("Batch")
ax.set_ylabel("Units_Produced (media)")
ax.grid(True)
ax.legend(title="Production_Line")
plt.show()




# Figura 12, 13, 14 y 15
fig, axes = plt.subplots(2, 2, figsize=(12,10))


# Figura 12
for s, st in shift_styles.items():
    sns.kdeplot(df.loc[df["Shift"]==s, "Temperature_C"], ax=axes[0,0],
                linestyle=st["linestyle"], color=st["color"], label=s)
axes[0,0].set_title("Fig. 12: Densidad de Temperature_C por Shift")
axes[0,0].grid(True)
axes[0,0].legend(title="Shift")


# Figura 13
df["Batch"] = (df["Product_ID"]-1)//20 + 1
pivot = df.groupby(["Batch","Shift"])["Units_Produced"].sum().unstack().fillna(0)
pivot.plot.area(ax=axes[0,1], color=[shift_styles[s]["color"] for s in pivot.columns])
axes[0,1].set_title("Fig. 13: Área apilada Units_Produced por Shift (Batch)")
axes[0,1].grid(True)


# Figura 14
counts = df.groupby(["Production_Line","Shift"]).size().unstack().fillna(0)
percent = counts.div(counts.sum(axis=1), axis=0)*100
percent.plot.bar(stacked=True, ax=axes[1,0], edgecolor="black",
                 color=[shift_styles[s]["color"] for s in percent.columns])
axes[1,0].set_title("Fig. 14: Composición de Turnos por Línea (%)")
axes[1,0].yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
axes[1,0].grid(True)
axes[1,0].legend(title="Shift")


# Figura 15
for line, data in df.groupby("Production_Line"):
    axes[1,1].scatter(data["Units_Produced"], data["Revenue_USD"],
                      color=line_colors[line], marker=line_markers[line],
                      label=line, alpha=0.7, edgecolor="black")
axes[1,1].set_title("Fig. 15: Units_Produced vs Revenue_USD (por línea)")
axes[1,1].set_xlabel("Units_Produced")
axes[1,1].set_ylabel("Revenue_USD")
axes[1,1].get_yaxis().set_major_formatter(lambda v, _: f"{int(v):,}")
axes[1,1].legend(title="Production_Line")
axes[1,1].grid(True)


plt.tight_layout()
plt.show()
