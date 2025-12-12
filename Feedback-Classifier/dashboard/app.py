import os
import json
import sqlite3

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/feedback/submit")

# ⚙️ Config general
st.set_page_config(
    page_title="Feedback Classifier",
    page_icon="💬",
    layout="wide",
)

# 🎨 Estilos
st.markdown(
    """
    <style>
    .metric-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, #020617, #020817);
        border: 1px solid #1f2937;
        margin-bottom: 0.6rem;
        box-shadow: 0 8px 20px rgba(15,23,42,0.6);
    }
    .metric-title {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: .12em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.25rem;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paletas para los gráficos
SENTIMENT_COLORS = ["#22c55e", "#eab308", "#ef4444", "#6366f1"]  # pos, neutro, neg, otro
URGENCY_COLORS = ["#22c55e", "#eab308", "#ef4444"]               # baja, media, alta
CATEGORY_COLORS = px.colors.qualitative.Set2
TOPIC_COLORS = px.colors.qualitative.Vivid


# -------------- UTILIDADES DB ----------------
def load_feedback_df() -> pd.DataFrame:
    """Carga todos los registros de feedback desde SQLite."""
    if not os.path.exists("feedback.db"):
        return pd.DataFrame()

    conn = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df


# -------------- SIDEBAR: TEST API --------------
def test_api_connection():
    st.sidebar.markdown("## 🔍 Test API")

    if st.sidebar.button("Probar Conexión"):
        try:
            payload = {"text": "ping", "source_id": 1, "external_id": "ping"}
            r = requests.post(API_URL, json=payload, timeout=10)

            if r.status_code == 200:
                st.sidebar.success("✅ API conectada y respondiendo OK")
            else:
                st.sidebar.warning(f"⚠️ API respondió con código {r.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ No se pudo conectar: {e}")


# -------------- PESTAÑA 1: CARGA MANUAL --------------
def tab_carga_manual():
    st.title("📊 Carga Manual de Feedback")
    st.caption("Clasifica y guarda mensajes provenientes de WhatsApp, formularios web, encuestas u otros canales.")
    st.markdown("---")

    col_form, col_result = st.columns([1.1, 1.3])

    with col_form:
        st.subheader("✉️ Nuevo feedback")

        with st.form(key="feedback_form"):
            message_text = st.text_area(
                "Mensaje de Feedback:",
                height=200,
                key="message_input",
                placeholder="Escribe o pega aquí el mensaje de WhatsApp, comentario de encuesta, etc.",
            )

            source = st.selectbox(
                "Fuente del Mensaje:",
                options=["WhatsApp", "Formulario Web", "Encuesta", "Otro"],
                key="source_select",
            )

            submit_button = st.form_submit_button(label="🚀 Clasificar y Guardar Feedback")

    with col_result:
        st.subheader("🧠 Resultados de la Clasificación")

    if submit_button:
        if not message_text.strip():
            st.error("Por favor, introduce un mensaje de feedback.")
            return

        source_mapping = {
            "WhatsApp": 1,
            "Formulario Web": 2,
            "Encuesta": 3,
            "Otro": 4,
        }

        payload = {
            "text": message_text,
            "source_id": source_mapping[source],
            "external_id": message_text[:20],
        }

        try:
            with st.spinner("Procesando feedback..."):
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"},
                )

            with col_result:
                with st.expander("🔍 Detalles de la respuesta (debug)", expanded=False):
                    st.write(f"**Status Code:** {response.status_code}")
                    st.write(f"**Response Headers:** {dict(response.headers)}")
                    st.write(f"**Raw Response (inicio):** `{response.text[:300]}...`")

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Error decodificando JSON: {e}")
                        st.text(f"Respuesta raw: {response.text}")
                        return

                    st.success("✅ ¡Feedback guardado con éxito!")

                    # Normalizamos estructura de respuesta
                    item = data[0] if isinstance(data, list) else data
                    analysis_wrapper = item.get("analysis", {})
                    inner = analysis_wrapper.get("analysis", analysis_wrapper)

                    sentiment = inner.get("sentiment", {})
                    category = inner.get("category", {})
                    urgency = inner.get("urgency", {})
                    topics = inner.get("topics", {})

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Sentimiento</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{sentiment.get("label", "-")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Categoría</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{category.get("label", "-")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Urgencia</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{urgency.get("label", "-")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Topic</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{topics.get("label", "-")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("### 🔎 JSON completo de la clasificación")
                    st.json(item)

                elif response.status_code == 422:
                    st.error("❌ Error de validación de datos")
                    st.code(response.text)
                else:
                    st.error(f"❌ Error del servidor: {response.status_code}")
                    st.code(response.text)

        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Error de conexión: {e}")
            st.info("Verifica que la API esté funcionando en: " + API_URL)
        except requests.exceptions.Timeout:
            st.error("⏰ Timeout: La API tardó más de 30 segundos en responder")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")


# -------------- PESTAÑA 2: DASHBOARD --------------
def tab_dashboard():
    st.title("📈 Dashboard de Feedback")
    st.caption("Visualización de métricas clave del feedback clasificado.")
    st.markdown("---")

    df = load_feedback_df()

    if df.empty:
        st.info("Aún no hay datos en la base de datos. Carga algunos feedbacks primero.")
        return

    # Normalizar fechas si existen
    if "analyzed_at" in df.columns:
        df["analyzed_at"] = pd.to_datetime(df["analyzed_at"], errors="coerce")
        df["fecha"] = df["analyzed_at"].dt.date
    else:
        df["fecha"] = None

    # ---------- FILTROS ----------
    st.markdown("### 🎛️ Filtros")

    col_f1, col_f2, col_f3 = st.columns(3)
    col_f4, col_f5 = st.columns(2)

    # Fecha
    if df["fecha"].notna().any():
        min_date = df["fecha"].min()
        max_date = df["fecha"].max()
        with col_f1:
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
    else:
        date_range = None

    # Sentimiento
    unique_sentiments = (
        sorted(df["sentiment_label"].dropna().unique())
        if "sentiment_label" in df.columns
        else []
    )
    with col_f2:
        selected_sentiments = st.multiselect(
            "Sentimiento",
            options=unique_sentiments,
            default=unique_sentiments,
        )

    # Urgencia
    unique_urgencies = (
        sorted(df["urgency_label"].dropna().unique())
        if "urgency_label" in df.columns
        else []
    )
    with col_f3:
        selected_urgencies = st.multiselect(
            "Urgencia",
            options=unique_urgencies,
            default=unique_urgencies,
        )

    # Categoría
    unique_categories = (
        sorted(df["category_label"].dropna().unique())
        if "category_label" in df.columns
        else []
    )
    with col_f4:
        selected_categories = st.multiselect(
            "Categoría",
            options=unique_categories,
            default=unique_categories,
        )

    # Fuente (source_id)
    unique_sources = (
        sorted(df["source_id"].dropna().unique())
        if "source_id" in df.columns
        else []
    )
    with col_f5:
        selected_sources = st.multiselect(
            "Fuente (ID)",
            options=unique_sources,
            default=unique_sources,
        )

    # ---------- APLICAR FILTROS ----------
    df_filtered = df.copy()

    # Filtro por fecha
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            df_filtered = df_filtered[
                (df_filtered["fecha"] >= start_date) & (df_filtered["fecha"] <= end_date)
            ]

    # Filtro por sentimiento
    if unique_sentiments and selected_sentiments:
        df_filtered = df_filtered[
            df_filtered["sentiment_label"].isin(selected_sentiments)
        ]

    # Filtro por urgencia
    if unique_urgencies and selected_urgencies:
        df_filtered = df_filtered[
            df_filtered["urgency_label"].isin(selected_urgencies)
        ]

    # Filtro por categoría
    if unique_categories and selected_categories:
        df_filtered = df_filtered[
            df_filtered["category_label"].isin(selected_categories)
        ]

    # Filtro por fuente
    if unique_sources and selected_sources:
        df_filtered = df_filtered[df_filtered["source_id"].isin(selected_sources)]

    st.markdown(
        f"**Registros filtrados:** {len(df_filtered)} / {len(df)} totales",
    )

    if df_filtered.empty:
        st.warning("No hay registros que coincidan con los filtros seleccionados.")
        return

    # ---------- KPIs ----------
    total = len(df_filtered)
    por_sentimiento = (
        df_filtered["sentiment_label"].value_counts()
        if "sentiment_label" in df_filtered.columns
        else pd.Series()
    )
    por_urgencia = (
        df_filtered["urgency_label"].value_counts()
        if "urgency_label" in df_filtered.columns
        else pd.Series()
    )
    por_categoria = (
        df_filtered["category_label"].value_counts()
        if "category_label" in df_filtered.columns
        else pd.Series()
    )
    por_topic = (
        df_filtered["topic_label"].value_counts()
        if "topic_label" in df_filtered.columns
        else pd.Series()
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total de feedbacks", total)
    with c2:
        if not por_sentimiento.empty:
            st.metric("Sentimiento top", por_sentimiento.idxmax())
        else:
            st.metric("Sentimiento top", "-")
    with c3:
        if not por_urgencia.empty:
            st.metric("Urgencia top", por_urgencia.idxmax())
        else:
            st.metric("Urgencia top", "-")
    with c4:
        if not por_categoria.empty:
            st.metric("Categoría top", por_categoria.idxmax())
        else:
            st.metric("Categoría top", "-")

    # ---------- GRÁFICOS ----------
    # Fila 1: donuts de sentimiento y urgencia
    col_pie_sent, col_pie_urg = st.columns(2)

    if not por_sentimiento.empty:
        with col_pie_sent:
            st.markdown("#### Distribución de Sentimientos")
            sent_df = por_sentimiento.reset_index()
            sent_df.columns = ["sentiment", "count"]
            fig_sent = px.pie(
                sent_df,
                values="count",
                names="sentiment",
                hole=0.45,
                color="sentiment",
                color_discrete_sequence=SENTIMENT_COLORS,
            )
            fig_sent.update_layout(
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    if not por_urgencia.empty:
        with col_pie_urg:
            st.markdown("#### Distribución de Urgencias")
            urg_df = por_urgencia.reset_index()
            urg_df.columns = ["urgency", "count"]
            fig_urg = px.pie(
                urg_df,
                values="count",
                names="urgency",
                hole=0.45,
                color="urgency",
                color_discrete_sequence=URGENCY_COLORS,
            )
            fig_urg.update_layout(
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_urg, use_container_width=True)

    # Fila 2: categorías (barras) + tendencia temporal (línea)
    col_cat, col_time = st.columns(2)

    if not por_categoria.empty:
        with col_cat:
            st.markdown("#### Top categorías")
            cat_df = por_categoria.reset_index()
            cat_df.columns = ["category", "count"]
            fig_cat = px.bar(
                cat_df,
                x="category",
                y="count",
                color="category",
                color_discrete_sequence=CATEGORY_COLORS,
            )
            fig_cat.update_layout(
                xaxis_title="",
                yaxis_title="Cantidad",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_time:
        st.markdown("#### Tendencia de volumen en el tiempo")
        if df_filtered["fecha"].notna().any():
            time_df = (
                df_filtered.groupby("fecha")["id"]
                .count()
                .reset_index()
                .rename(columns={"id": "cantidad"})
            )
            fig_time = px.line(
                time_df,
                x="fecha",
                y="cantidad",
                markers=True,
                color_discrete_sequence=["#6366f1"],
            )
            fig_time.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Cantidad",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Aún no hay fechas de análisis registradas para construir la serie de tiempo.")

    # Fila 3: topics (si existen)
    if not por_topic.empty:
        st.markdown("#### Distribución de Topics detectados")
        topic_df = por_topic.reset_index()
        topic_df.columns = ["topic", "count"]
        fig_topic = px.bar(
            topic_df,
            x="topic",
            y="count",
            color="topic",
            color_discrete_sequence=TOPIC_COLORS,
        )
        fig_topic.update_layout(
            xaxis_title="",
            yaxis_title="Cantidad",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#020617",
            font=dict(color="#e5e7eb"),
        )
        st.plotly_chart(fig_topic, use_container_width=True)

    # Tabla detalle
    st.markdown("### Registros crudos (después de filtros)")
    st.dataframe(df_filtered.sort_values("id", ascending=False))


# -------------- MAIN --------------
def main():
    test_api_connection()
    tab1, tab2 = st.tabs(["✉️ Carga Manual", "📊 Dashboard"])

    with tab1:
        tab_carga_manual()

    with tab2:
        tab_dashboard()


if __name__ == "__main__":
    main()
