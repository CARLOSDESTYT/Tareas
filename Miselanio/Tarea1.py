import pandas as pd
import numpy as np
import random


# 1. Importación y exploración
#   A. Carga el archivo basquet_100.csv en un DataFrame.
#   B. Revisa las dimensiones de la tabla (cuántas filas y columnas).
#   C. Identifica el tipo de datos de cada columna.
df = pd.read_csv("basquet_100.csv")
print("Dimensiones:", df.shape)
print("Tipos de datos:\n", df.dtypes)


# 2. Limpieza de datos
# A. Elige 50 posiciones aleatorias dentro de la tabla y reemplazarlas con valores nulos (NaN). Asegúrate de que los valores nulos se distribuyan entre diferentes filas y columnas, no concentrados en un solo lugar.
# B. Cuenta la cantidad de valores nulos por columna y por fila
# C. Reglas de limpieza
# D. Verificar nulos
for _ in range(50):
    fila = random.randint(0, df.shape[0]-1)
    col = random.choice(df.columns)
    df.at[fila, col] = np.nan

print("Nulos por columna:\n", df.isna().sum())
print("Nulos por fila:\n", df.isna().sum(axis=1))

for col in df.columns:
    if df[col].dtype == "object":
        df = df.dropna(subset=[col])
    else:
        df[col] = df[col].fillna(df[col].mean())

print("Nulos después de limpiar:", df.isna().sum().sum())


# 3. Filtrado y selección
#   A. Selecciona todos los jugadores que anotaron más de 15 puntos.
#   B. Selecciona los jugadores del equipo A que tuvieron más de 5 asistencias.
#   C. Encuentra los jugadores que lograron al menos 10 rebotes.
jugadores_15p = df[df["Puntos"] > 15]
jugadores_equipoA_5a = df[(df["Equipo"] == "A") & (df["Asistencias"] > 5)]
jugadores_10r = df[df["Rebotes"] >= 10]


# 4. Columnas nuevas
#   A. Crea una columna llamada Eficiencia, definida como: Eficiencia = Puntos + Asistencias + Rebotes
#   B. Agrega una nueva columna llamada Turno, en la que cada jugador sea asignado leatoriamente como Matutino o Vespertino.
df["Eficiencia"] = df["Puntos"] + df["Asistencias"] + df["Rebotes"]
df["Turno"] = np.random.choice(["Matutino", "Vespertino"], size=len(df))


# 5. Agrupaciones
#   A. Calcula el promedio de puntos por equipo.
#   B. Calcula la cantidad máxima de puntos por equipo.
#   C. Calcula el promedio de eficiencia por turno (matutino vs. vespertino).
#   D. Obtén el número de jugadores por equipo y por turno al mismo tiempo.
print("Promedio puntos por equipo:\n", df.groupby("Equipo")["Puntos"].mean())
print("Máx puntos por equipo:\n", df.groupby("Equipo")["Puntos"].max())
print("Promedio eficiencia por turno:\n", df.groupby("Turno")["Eficiencia"].mean())
print("Jugadores por equipo y turno:\n", df.groupby(["Equipo", "Turno"]).size())


# 6. Ordenamiento y top-N
df_sorted = df.sort_values(by="Eficiencia", ascending=False)
top5_por_equipo = df_sorted.groupby("Equipo").head(5)
jugador_max_rebotes = df.loc[df["Rebotes"].idxmax()]


# 7. Permutaciones de equipos
equipos = df["Equipo"].unique()
combinaciones = [(a,b) for i,a in enumerate(equipos) for b in equipos[i+1:]]
promedios = df.groupby("Equipo")["Puntos"].mean().to_dict()

tabla_duelos = pd.DataFrame([{
    "Equipo_1": a,
    "Equipo_2": b,
    "Promedio_Equipo_1": promedios[a],
    "Promedio_Equipo_2": promedios[b]
} for a,b in combinaciones])


# 8. Exportación
df_sorted.head(10).to_csv("top10_eficiencia.csv", index=False)
tabla_duelos.to_csv("duelos_equipos.csv", index=False)
