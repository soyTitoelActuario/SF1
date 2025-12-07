import Funciones as f
import streamlit as st
import pandas as pd
import numpy as np
#primero hago lo que se pide de laws dos estategias
#regiones del mundo y sectores de estados unidos, ademas tengo los benchmarks

# Configuración de la página
 # Crear pestañas
st.set_page_config(layout="wide")   
tab1, tab2,tab3 = st.tabs(["Análisis de Activos (Portafolio arbitrario)", "Análisis de Activos (Portafolio Optimizado)","Análisis de Activos (Portafolio Black-Litterman)"])
with tab1:
    st.title("📊 Dashboard de Análisis de Inversión")
    # Definir benchmarks
    benchmark_options = {
        "Regiones del Mundo": {
            'SPLG': 70.62,
            'EWC': 3.23,
            'IEUR': 11.76,
            'EEM': 9.02,
            'EWJ': 5.37},
        "Sectores de Estados Unidos": {
            'XLC': 9.99,
            'XLY': 10.25,
            'XLP': 4.82,
            'XLE': 2.95,
            'XLF': 13.07,
            'XLV': 9.58,
            'XLI': 8.09,
            'XLB': 1.66,
            'XLRE': 1.87,
            'XLK': 35.35,
            'XLU': 2.37}}
    # Selector en el sidebar
    selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark dada tu estrategia:", list(benchmark_options.keys()))
    st.sidebar.markdown("---")
    # Entrada de Tickers y pesos
    simbolos_input = st.sidebar.text_input("Ingrese los Tickers de los instrumentos separados por comas:",
        "NVDA, AAPL, GOOGL, TSLA",help=f"Debe corresponder a la estrategia: {selected_benchmark}")
    pesos_input = st.sidebar.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):",
        "0.3, 0.2, 0.2, 0.3")
    # Configurar fechas
    start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2025-11-08"))
    # Procesar inputs
    simbolos = [s.strip() for s in simbolos_input.split(',')]
    pesos = [float(w.strip()) for w in pesos_input.split(',')]
    # Obtener el benchmark seleccionado
    benchmark = benchmark_options[selected_benchmark]
    st.subheader(f"Estrategia: {selected_benchmark}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Benchmark")
        df_benchmark = pd.DataFrame(list(benchmark.items()), columns=['ETF', 'Peso (%)'])
        df_benchmark = df_benchmark.sort_values('Peso (%)', ascending=False)
        st.dataframe(df_benchmark, use_container_width=True, hide_index=True)
        st.metric("Total", f"{df_benchmark['Peso (%)'].sum():.2f}%")

    with col2:
        st.markdown("### 💼 Tu Portafolio")
        df_portafolio = pd.DataFrame({'Ticker': simbolos, 'Peso (%)': [p * 100 for p in pesos]})
        st.dataframe(df_portafolio, use_container_width=True, hide_index=True)
        st.metric("Total", f"{sum(pesos) * 100:.2f}%")
        # Validación
        if abs(sum(pesos) - 1.0) > 0.01:
            st.warning("⚠️ Los pesos deben sumar 1.0")

    # Análisis con IA del Portafolio
    st.markdown("---")
    st.subheader("🤖 Análisis del Portafolio con IA")
    # Botón para generar análisis con IA
    if st.button("📊 Generar análisis de tu portafolio con IA"):
        with st.spinner("Consultando análisis con IA..."):
            tickers_str = ", ".join(simbolos)
            prompt = f"""
            Proporciona una breve descripción de los siguientes activos: {tickers_str}
            Para cada uno menciona:
            - Qué tipo de activo es
            - Sector o región que representa
            Máximo 150 palabras, estilo conciso.
            """
            analisis_ia = f.preguntar_groq(prompt)
            st.write(analisis_ia)

    # Sección para preguntas específicas
    with st.expander("💬 Hacer pregunta específica sobre tu portafolio"):
        pregunta_especifica = st.text_input(
            "Escribe tu pregunta sobre el portafolio:",
            key="pregunta_portafolio"
        )
        if st.button("Enviar pregunta", key="btn_pregunta_portafolio"):
            if pregunta_especifica:
                with st.spinner("Buscando respuesta..."):
                    tickers_str = ", ".join(simbolos)
                    respuesta = f.preguntar_groq(f"Sobre el portafolio compuesto por {tickers_str}: {pregunta_especifica}")
                    st.write("**Respuesta:**")
                    st.write(respuesta)
            else:
                st.warning("Por favor escribe una pregunta")
    
    # Obtener datos del portafolio del usuario
    datos_portafolio = f.obtener_datos_acciones(simbolos, start_date, end_date)
    returns_portafolio, cumulative_returns_portafolio, normalized_prices_portafolio = f.calcular_metricas(datos_portafolio)
    portfolio_returns = f.calcular_rendimientos_portafolio(returns_portafolio, pesos)

    # Obtener datos del benchmark
    benchmark_tickers = list(benchmark.keys())
    benchmark_weights = [v/100 for v in benchmark.values()]  # Convertir de % a decimal
    datos_benchmark = f.obtener_datos_acciones(benchmark_tickers, start_date, end_date)
    returns_benchmark, cumulative_returns_benchmark, normalized_prices_benchmark = f.calcular_metricas(datos_benchmark)
    benchmark_returns = f.calcular_rendimientos_portafolio(returns_benchmark, benchmark_weights)
    cum_ret_port = (1 + portfolio_returns).cumprod()
    var_95_port, cvar_95_port = f.calcular_var_cvar(portfolio_returns)
    calmar_port = f.calmar_ratio(portfolio_returns) # Devuelve una tupla o float? Tu funcion devuelve (calmar, cagr) o solo calmar? 
    # REVISION: Tu funcion calmar_ratio devuelve solo 'calmar'. Perfecto.
    
    metricas_portafolio = {
        'Rendimiento': portfolio_returns.mean() * 252,
        'Volatilidad': portfolio_returns.std() * np.sqrt(252),
        'Beta': f.calcular_beta_vs_sp500(portfolio_returns, start_date, end_date),
        'Sharpe': f.sharpe_ratio(portfolio_returns),
        'Sortino': f.sortino_ratio(portfolio_returns),
        'Treynor': f.treynor_ratio(portfolio_returns, benchmark_returns),
        'Info Ratio': f.information_ratio(portfolio_returns, benchmark_returns),
        'Calmar': f.calmar_ratio(portfolio_returns),
        'VaR': var_95_port,
        'CVaR': cvar_95_port,
        'Max Drawdown': f.calcular_max_drawdown(cum_ret_port),
        'Sesgo': f.calcular_skewness(portfolio_returns),
        'Curtosis': f.calcular_kurtosis(portfolio_returns)}

    cum_ret_bench = (1 + benchmark_returns).cumprod()
    var_95_bench, cvar_95_bench = f.calcular_var_cvar(benchmark_returns)
    beta_bench = f.calcular_beta_vs_sp500(benchmark_returns, start_date, end_date)

    metricas_benchmark = {
        'Rendimiento': benchmark_returns.mean() * 252,
        'Volatilidad': benchmark_returns.std() * np.sqrt(252),
        'Beta': beta_bench,
        'Sharpe': f.sharpe_ratio(benchmark_returns),
        'Sortino': f.sortino_ratio(benchmark_returns),
        'Treynor': f.treynor_ratio(benchmark_returns, benchmark_returns), # Treynor vs sí mismo
        'Info Ratio': 0.0, # El Info Ratio del benchmark contra sí mismo es 0 por definición
        'Calmar': f.calmar_ratio(benchmark_returns),
        'VaR': var_95_bench,
        'CVaR': cvar_95_bench,
        'Max Drawdown': f.calcular_max_drawdown(cum_ret_bench),
        'Sesgo': f.calcular_skewness(benchmark_returns),
        'Curtosis': f.calcular_kurtosis(benchmark_returns)
    }
    st.markdown("---")
    st.subheader("📊 Performance Integral: Portafolio vs Benchmark")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rendimiento Anual", f"{metricas_portafolio['Rendimiento']:.2%}", delta=f"{metricas_portafolio['Rendimiento'] - metricas_benchmark['Rendimiento']:.2%}")
    with col2:
        st.metric("Volatilidad", f"{metricas_portafolio['Volatilidad']:.2%}", delta=f"{metricas_portafolio['Volatilidad'] - metricas_benchmark['Volatilidad']:.2%}", delta_color="inverse")
    with col3:
        st.metric("Beta (vs SP500)", f"{metricas_portafolio['Beta']:.2f}", delta=f"{metricas_portafolio['Beta'] - metricas_benchmark['Beta']:.2f}", delta_color="off")
    with col4:
        st.metric("Max Drawdown", f"{metricas_portafolio['Max Drawdown']:.2%}", delta=f"{metricas_portafolio['Max Drawdown'] - metricas_benchmark['Max Drawdown']:.2%}", delta_color="inverse")

    st.markdown("##### 🧠 Ratios de Eficiencia Ajustados por Riesgo")
    col5, col6, col7, col8, col9 = st.columns(5)  # ✅ 5 columnas

    with col5:
        st.metric("Sharpe", f"{metricas_portafolio['Sharpe']:.2f}", delta=f"{metricas_portafolio['Sharpe'] - metricas_benchmark['Sharpe']:.2f}")
    with col6:
        st.metric("Sortino", f"{metricas_portafolio['Sortino']:.2f}", delta=f"{metricas_portafolio['Sortino'] - metricas_benchmark['Sortino']:.2f}")
    with col7:
        st.metric("Treynor", f"{metricas_portafolio['Treynor']:.2f}", delta=f"{metricas_portafolio['Treynor'] - metricas_benchmark['Treynor']:.2f}")
    with col8:
        st.metric("Info Ratio", f"{metricas_portafolio['Info Ratio']:.2f}",  delta=f"{metricas_portafolio['Info Ratio'] - metricas_benchmark['Info Ratio']:.2f}")
    with col9:
        st.metric("Calmar", f"{metricas_portafolio['Calmar']:.2f}", delta=f"{metricas_portafolio['Calmar'] - metricas_benchmark['Calmar']:.2f}")

    st.markdown("##### 🛡️ Métricas de Riesgo de Cola")
    col10, col11, col12, col13 = st.columns(4)

    with col10:
        st.metric("VaR (95% método Historico)", f"{metricas_portafolio['VaR']:.2%}", delta=f"{metricas_portafolio['VaR'] - metricas_benchmark['VaR']:.2%}", delta_color="inverse")
    with col11:
        st.metric("CVaR (95% y método Historico)", f"{metricas_portafolio['CVaR']:.2%}", delta=f"{metricas_portafolio['CVaR'] - metricas_benchmark['CVaR']:.2%}", delta_color="inverse")
    with col12:
        st.metric("Sesgo", f"{metricas_portafolio['Sesgo']:.2f}", 
                delta=f"{metricas_portafolio['Sesgo'] - metricas_benchmark['Sesgo']:.2f}")
    with col13:
        st.metric("Curtosis", f"{metricas_portafolio['Curtosis']:.2f}", delta=f"{metricas_portafolio['Curtosis'] - metricas_benchmark['Curtosis']:.2f}", delta_color="inverse")
    with st.expander("📋 Ver Desglose Completo (Tabla)"):
        df_comparacion = pd.DataFrame({
            'Tu Portafolio': metricas_portafolio,
            'Benchmark': metricas_benchmark,
            'Diferencia': {k: metricas_portafolio[k] - metricas_benchmark[k] for k in metricas_portafolio}
        })
        st.dataframe(df_comparacion.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1))

with tab2:
    st.subheader("📈 Análisis de Portafolio Optimizado")
    datos_opt = f.obtener_datos_acciones(simbolos, start_date, end_date)
    # 2. Calcular retornos GEOMÉTRICOS 
    def calcular_expected_returns_geometrico(prices):
        n = len(prices)
        total_return = (prices.iloc[-1] / prices.iloc[0])
        n_periods = n / 252  # Años
        expected_returns = total_return ** (1 / n_periods) - 1
        return expected_returns
    expected_returns_temp = calcular_expected_returns_geometrico(datos_opt)
    expected_returns = expected_returns_temp[simbolos]

    # 3. Matriz de Covarianza
    returns_opt = datos_opt.pct_change().dropna()
    cov_matrix_temp = returns_opt.cov() * 252
    cov_matrix = cov_matrix_temp.loc[simbolos, simbolos]
    col_config1, col_config2 = st.columns([1, 2])
    
    with col_config1:
        st.markdown("##### Configuración")
        
        tipo_optimizacion = st.radio(
            "Seleccione objetivo:",
            ["Mínima Varianza", "Máximo Sharpe", "Rendimiento Objetivo"]
        )
        
        rendimiento_objetivo = None
        if tipo_optimizacion == "Rendimiento Objetivo":
            rendimiento_objetivo = st.number_input(
                "Rendimiento anual deseado:",
                min_value=0.0, 
                max_value=2.0, 
                value=0.15, 
                step=0.01,
                format="%.2f",
                help="Ej: 0.15 para 15%"
            )
            
        boton_optimizar = st.button("⚡ Calcular Optimización", use_container_width=True)
    
    with col_config2:
        # Espacio para resultados
        if boton_optimizar:
            with st.spinner("Ejecutando algoritmos de optimización..."):
                try:
                    if tipo_optimizacion == "Mínima Varianza":
                        pesos_opt, (ret_opt, vol_opt) = f.minima_varianza(expected_returns.values, cov_matrix.values)
                    
                    elif tipo_optimizacion == "Máximo Sharpe":
                        # Asumimos risk_free=0.0 por defecto o puedes pasar rf
                        pesos_opt, (ret_opt, vol_opt) = f.maximo_sharpe(expected_returns.values, cov_matrix.values)
                    
                    else: # Markowitz Target
                        pesos_opt, (ret_opt, vol_opt) = f.markowitz_rendimiento_objetivo(
                            expected_returns.values, 
                            cov_matrix.values, 
                            rendimiento_objetivo
                        )

                    # --- MOSTRAR RESULTADOS ---
                    st.success(f"✅ Optimización Exitosa: {tipo_optimizacion}")
                    
                    # 1. Métricas del Portafolio Optimizado
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rendimiento Esperado", f"{ret_opt:.2%}")
                    c2.metric("Volatilidad Esperada", f"{vol_opt:.2%}")
                    c3.metric("Sharpe Ratio", f"{(ret_opt/vol_opt):.2f}")

                    # 2. DataFrame de Pesos
                    df_optimizado = pd.DataFrame({
                        'Ticker': simbolos,
                        'Peso (%)': pesos_opt * 100
                    }).sort_values('Peso (%)', ascending=False)
                    df_optimizado = df_optimizado[df_optimizado['Peso (%)'] > 0.01]
                    col_tabla, col_grafico = st.columns([1, 1])
                    
                    with col_tabla:
                        st.markdown("###### Asignación de Activos")
                        st.dataframe(
                            df_optimizado.style.format({"Peso (%)": "{:.2f}%"}).background_gradient(cmap="Greens"),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_grafico:
                        import plotly.express as px
                        fig_pie = px.pie(
                            df_optimizado, 
                            values='Peso (%)', 
                            names='Ticker', 
                            title='Distribución Óptima',
                            hole=0.4
                        )
                        fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
                        st.plotly_chart(fig_pie, use_container_width=True)

                except Exception as e:
                    st.error(f"⚠️ No se pudo optimizar: {str(e)}")
                    st.info("Intenta con un rendimiento objetivo más bajo o verifica los datos.")
        else:
            st.info("👈 Selecciona un método y haz clic en 'Calcular' para ver la propuesta óptima.")

with tab3:
    st.title("Proximamante...")
