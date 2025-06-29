import pandas as pd
import pyomo.environ as pyo
from datetime import datetime
import os
import random

# --- 1. Carga de datos ---
datos = pd.read_excel('Datos.xlsx', sheet_name='Newman')#.head(100)

periodo = [1, 2, 3, 4, 5]
max_extr = 2000000
max_pros_dict = [1100000, 1100000, 1100000, 1100000, 1100000]
min_res = 7000
min_grad = 0.03

block = datos['id'].tolist()

# === AJUSTE ALEATORIO AL BENEFICIO ===
periodo_aleatorio = random.choice(periodo)
tipo_ajuste = random.choice(['penalizacion', 'bonificacion'])

if tipo_ajuste == 'penalizacion':
    porcentaje = random.uniform(0.05, 0.10)  # entre 5% y 10%
    factor = 1 - porcentaje
else:
    porcentaje = random.uniform(0.03, 0.08)  # entre 3% y 8%
    factor = 1 + porcentaje

columna_beneficio = f'Benefit P{periodo_aleatorio}'
datos[columna_beneficio] *= factor

print(f"\n>>> Se aplicó una {'penalización' if tipo_ajuste == 'penalizacion' else 'bonificación'} del {porcentaje:.2%} al beneficio del período {periodo_aleatorio}.")

# --- 2. Precedencias ---
adyacentes = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
predecesores = []
for idx, fila in datos.iterrows():
    x, y, z = fila['x'], fila['y'], fila['z']
    for dx, dy in adyacentes:
        filtro = (
            (datos['x'] == x + dx) &
            (datos['y'] == y + dy) &
            (datos['z'] == z + 1)
        )
        for _, bloque_sup in datos[filtro].iterrows():
            predecesores.append((fila['id'], bloque_sup['id']))

# --- 3. Modelo Pyomo ---
def construir_modelo(datos, periodo, block, max_extr, max_pros, min_grad, min_res, predecesores):
    model = pyo.ConcreteModel()
    model.block = pyo.Set(initialize=block)
    model.periodo = pyo.Set(initialize=periodo)
    model.precedencias = pyo.Set(initialize=predecesores, dimen=2)
    model.x = pyo.Var(model.block, model.periodo, domain=pyo.Binary)

    def FO(model):
        return sum(
            model.x[b, t] * datos.loc[datos['id'] == b, f'Benefit P{t}'].values[0]
            for b in model.block for t in model.periodo
        )
    model.FO = pyo.Objective(rule=FO, sense=pyo.maximize)

    def una_extraccion_por_bloque(model, b):
        return sum(model.x[b, t] for t in model.periodo) <= 1
    model.unica_extraccion = pyo.Constraint(model.block, rule=una_extraccion_por_bloque)

    def capacidad_extraccion(model, t):
        return sum(model.x[b, t] * datos.loc[datos['id'] == b, 'tonns'].values[0] for b in model.block) <= max_extr
    model.capacidad_extraccion = pyo.Constraint(model.periodo, rule=capacidad_extraccion)

    def capacidad_procesamiento(model, t):
        return sum(model.x[b, t] * datos.loc[datos['id'] == b, 'tonns'].values[0] for b in model.block) <= max_pros[t-1]
    model.capacidad_procesamiento = pyo.Constraint(model.periodo, rule=capacidad_procesamiento)

    def restriccion_precedencia(model, b, bp, t):
        return model.x[b, t] <= sum(model.x[bp, tau] for tau in model.periodo if tau <= t)
    model.precedencia = pyo.Constraint(model.precedencias, model.periodo, rule=restriccion_precedencia)

    def reserva_expuesta(model, t):
        return sum(
            (1 - sum(model.x[b, tau] for tau in model.periodo if tau <= t)) *
            datos.loc[datos['id'] == b, 'tonns'].values[0] *
            int(datos.loc[datos['id'] == b, 'grade'].values[0] >= min_grad)
            for b in model.block
        ) >= min_res
    model.reserva_expuesta = pyo.Constraint(model.periodo, rule=reserva_expuesta)

    return model

model = construir_modelo(datos, periodo, block, max_extr, max_pros_dict, min_grad, min_res, predecesores)

solver = pyo.SolverFactory('cbc', executable='C:\\Users\\ricar\\Downloads\\Cbc-releases.2.10.12-w64-msvc16-md\\bin\\cbc.exe')
solver.options['seconds'] = 3*3600
solver.options['threads'] = 12
resultado = solver.solve(model, tee=True)

# --- Cálculo reserva expuesta ---
def calcular_reserva_expuesta(t_actual):
    reserva_expuesta_bloques = []
    reserva_expuesta_toneladas = 0
    for b in block:
        if sum(pyo.value(model.x[b, tau]) for tau in periodo if tau <= t_actual) < 0.5:
            predecesores_b = [bp for (bi, bp) in predecesores if bi == b]
            predecesores_extraidos = all(
                sum(pyo.value(model.x[bp, tau]) for tau in periodo if tau <= t_actual) > 0.5
                for bp in predecesores_b
            )
            fila = datos.loc[datos['id'] == b].iloc[0]
            if predecesores_extraidos and fila['grade'] >= min_grad:
                reserva_expuesta_bloques.append(b)
                reserva_expuesta_toneladas += fila['tonns']
    return reserva_expuesta_bloques, reserva_expuesta_toneladas

# --- Postproceso de resultados ---
resultados = []
beneficio_acumulado = 0

for idx, t in enumerate(periodo):
    bloques_extraidos = [b for b in block if pyo.value(model.x[b, t]) > 0.5]
    a_procesamiento = []
    desechados = []
    beneficio_periodo = 0
    toneladas_extraidas = 0
    toneladas_procesamiento = 0
    toneladas_desechadas = 0

    for b in bloques_extraidos:
        fila = datos.loc[datos['id'] == b].iloc[0]
        grado = fila['grade']
        toneladas = fila['tonns']
        beneficio = fila[f'Benefit P{t}']
        toneladas_extraidas += toneladas
        if grado >= min_grad:
            a_procesamiento.append(b)
            toneladas_procesamiento += toneladas
            beneficio_periodo += beneficio
        else:
            desechados.append(b)
            toneladas_desechadas += toneladas

    beneficio_acumulado += beneficio_periodo
    t_sig = periodo[idx + 1] if idx < len(periodo) - 1 else t
    reserva_expuesta_bloques, reserva_expuesta_toneladas = calcular_reserva_expuesta(t_sig)

    resultados.append({
        "periodo": t,
        "bloques_extraidos": bloques_extraidos,
        "z_bloques_extraidos": [datos.loc[datos['id'] == b, 'z'].values[0] for b in bloques_extraidos],
        "procesamiento": a_procesamiento,
        "z_procesamiento": [datos.loc[datos['id'] == b, 'z'].values[0] for b in a_procesamiento],
        "desechados": desechados,
        "z_desechados": [datos.loc[datos['id'] == b, 'z'].values[0] for b in desechados],
        "reserva_expuesta": reserva_expuesta_bloques,
        "z_reserva_expuesta": [datos.loc[datos['id'] == b, 'z'].values[0] for b in reserva_expuesta_bloques],
        "toneladas_extraidas": toneladas_extraidas,
        "toneladas_procesamiento": toneladas_procesamiento,
        "toneladas_desechadas": toneladas_desechadas,
        "reserva_expuesta_toneladas": reserva_expuesta_toneladas,
        "beneficio_neto_acumulado": beneficio_acumulado
    })

tabla_resultados = []
for res in resultados:
    tabla_resultados.append({
        "Período": res['periodo'],
        "Bloques Extraídos (IDs)": ', '.join(map(str, res['bloques_extraidos'])),
        "N° Bloques Extraídos": len(res['bloques_extraidos']),
        "Z Bloques Extraídos": ', '.join(map(str, res['z_bloques_extraidos'])),
        "Toneladas Extraídas": res['toneladas_extraidas'],
        "Bloques a Procesamiento (IDs)": ', '.join(map(str, res['procesamiento'])),
        "N° Bloques a Procesamiento": len(res['procesamiento']),
        "Z Procesamiento": ', '.join(map(str, res['z_procesamiento'])),
        "Toneladas a Procesamiento": res['toneladas_procesamiento'],
        "Bloques Desechados (IDs)": ', '.join(map(str, res['desechados'])),
        "N° Bloques Desechados": len(res['desechados']),
        "Z Desechados": ', '.join(map(str, res['z_desechados'])),
        "Toneladas Desechadas": res['toneladas_desechadas'],
        "Bloques en Reserva Expuesta (IDs)": ', '.join(map(str, res['reserva_expuesta'])),
        "N° Bloques en Reserva Expuesta": len(res['reserva_expuesta']),
        "Z Reserva Expuesta": ', '.join(map(str, res['z_reserva_expuesta'])),
        "Tonelaje Reserva Expuesta (t)": res['reserva_expuesta_toneladas'],
        "Beneficio Neto Acumulado ($)": res['beneficio_neto_acumulado'],
    })
df_resultados = pd.DataFrame(tabla_resultados)

# --- Resumen final ---
bloques_extraidos_final = [b for b in block if sum(pyo.value(model.x[b, t]) for t in periodo) > 0.5]
bloques_no_extraidos_final = [b for b in block if sum(pyo.value(model.x[b, t]) for t in periodo) < 0.5]
beneficio_total_acumulado = resultados[-1]['beneficio_neto_acumulado'] if resultados else 0

df_bloques_final = pd.DataFrame({
    "Bloques Extraídos (IDs)": [', '.join(map(str, bloques_extraidos_final))],
    "Cantidad Total de Bloques Extraídos": [len(bloques_extraidos_final)],
    "Bloques Restantes por Extraer (IDs)": [', '.join(map(str, bloques_no_extraidos_final))],
    "Cantidad de Bloques Restantes": [len(bloques_no_extraidos_final)],
    "Beneficio Total Acumulado ($)": [beneficio_total_acumulado]
})

# --- Información del ajuste aplicado ---
df_ajuste = pd.DataFrame({
    "Tipo de Ajuste": [tipo_ajuste],
    "Período Ajustado": [periodo_aleatorio],
    "Porcentaje de Ajuste": [f"{porcentaje:.2%}"],
    "Factor Aplicado": [f"{factor:.5f}"]
})
nombre_hoja_ajuste = 'Ajuste_' + datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Exportar a Excel ---
nombre_archivo = 'Resultados_Optimizacion_random_3horas.xlsx'
nombre_hoja = 'Resultados_' + datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_hoja_final = 'Resumen_Final_' + datetime.now().strftime("%Y%m%d_%H%M%S")

if os.path.exists(nombre_archivo):
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
        df_resultados.to_excel(writer, sheet_name=nombre_hoja, index=False)
        df_bloques_final.to_excel(writer, sheet_name=nombre_hoja_final, index=False)
        df_ajuste.to_excel(writer, sheet_name=nombre_hoja_ajuste, index=False)
else:
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_resultados.to_excel(writer, sheet_name=nombre_hoja, index=False)
        df_bloques_final.to_excel(writer, sheet_name=nombre_hoja_final, index=False)
        df_ajuste.to_excel(writer, sheet_name=nombre_hoja_ajuste, index=False)

print(f"\n>>> Resultados exportados exitosamente a '{nombre_archivo}' en las hojas '{nombre_hoja}', '{nombre_hoja_final}' y '{nombre_hoja_ajuste}'.")
