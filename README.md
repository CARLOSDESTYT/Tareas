# Prácticas de minería de datos

Repositorio para documentar prácticas de exploración, limpieza y análisis de datos.

## Alcance actual

La primera práctica de referencia analiza el rendimiento de 100 jugadores de basquetbol a partir de puntos, asistencias y rebotes. Incluye limpieza de valores faltantes, resúmenes por equipo y un ranking de eficiencia individual.

## Problema y objetivo

- Los 100 registros incluyen puntos, asistencias y rebotes, pero cada estadística por separado no permite comparar rápidamente el rendimiento global.
- Sin limpiar y resumir esos datos, cuesta identificar a los jugadores destacados y comparar resultados entre equipos.

**Objetivo medible:** Calcular la eficiencia de los 100 jugadores como `Puntos + Asistencias + Rebotes` y obtener los cinco jugadores con mayor eficiencia de cada equipo.

**Siguiente paso:** Ejecutar la limpieza de `basquet_100.csv`, comprobar cuántos registros quedan válidos y revisar el top 5 de cada equipo antes de exportar los resultados.
