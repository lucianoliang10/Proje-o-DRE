import pandas as pd
import streamlit as st

BASE_YEAR = 2026
PROJECTION_YEARS = list(range(2027, 2037))
ALL_YEARS = [BASE_YEAR] + PROJECTION_YEARS


GROUP_DEFINITIONS = [
    {
        "name": "Gross operating revenue",
        "children": [
            "Broadcast Revenue",
            "Matchday",
            "Merit Award",
            "Commercial",
            "Transfer market",
            "Space Lease",
            "Camisa 7 Program",
            "Camisa 6 Program",
            "Sale of Products",
            "Other Revenues",
        ],
    },
    {
        "name": "(–) Revenue deductions",
        "children": [
            "Direito De Arena",
            "Tef - Tributação Específica Do Futebol",
            "Vendas Canceladas",
            "Icms Sobre Vendas",
            "Devolução De Vendas",
            "Icms Difal Sobre Vendas",
            "Comissão Sobre Vendas",
            "Deduções De Vendas (Frete/Gateway)",
            "Dedução De Remuneração Sobre Contratos",
            "Impostos Internacionais",
        ],
    },
    {
        "name": "(–) Costs",
        "children": [
            "Football Payroll Costs",
            "Prize",
            "Logistics",
            "Championship Cost",
            "Costs of Goods Sold",
            "CT",
            "Costs",
            "Football Costs",
            "Player Transfer Costs",
        ],
    },
    {
        "name": "SG&A",
        "children": [
            "Payroll",
            "Prize expenses",
            "Supplies Expense",
            "Power & Utilities",
            "External Services",
            "General expenses",
            "Registration Rights",
            "Match Expenses",
            "Sales Expenses",
            "Equity",
            "Contingencies",
        ],
    },
    {
        "name": "Write-downs of intangible",
        "children": [],
    },
    {
        "name": "Depreciation and Amortization",
        "children": [
            "Depreciação De Veículos",
            "Depreciação Máquinas E Equipamentos",
            "Depreciação De Moveis E Utensilios",
            "Depreciação De Equipamentos Informática",
            "Amortização Da Benfeitoria",
            "Amortização De Atletas",
            "Amortização De Programas E Softwares",
            "Amortização De Direito De Uso",
            "Amortização De Atletas Profissionais",
            "Depreciação De Veículos",
            "Depreciação Máquinas E Equipamentos",
            "Amortização De Programas E Softwares",
            "Baixa De Ativo Imobilizado",
        ],
    },
    {
        "name": "Net finance income (expense)",
        "children": [
            "Financial Revenue",
            "Financial Expenses",
            "Unrealized FX Variation",
            "Shareholders Agreement",
            "Taxes",
        ],
    },
]

CALC_LINES = [
    {
        "name": "Net operating revenue",
        "formula": ["Gross operating revenue", "(–) Revenue deductions"],
    },
    {
        "name": "Gross profit / loss",
        "formula": ["Net operating revenue", "(–) Costs"],
    },
    {
        "name": "EBITDA",
        "formula": ["Gross profit / loss", "SG&A"],
    },
    {
        "name": "Adjusted EBITDA",
        "formula": ["EBITDA", "Write-downs of intangible"],
    },
    {
        "name": "EBIT",
        "formula": ["Adjusted EBITDA", "Depreciation and Amortization"],
    },
    {
        "name": "Profit/Loss",
        "formula": ["EBIT", "Net finance income (expense)"],
    },
]


def normalize_columns(df: pd.DataFrame) -> dict:
    return {col.lower().strip(): col for col in df.columns}


def detect_default_columns(df: pd.DataFrame) -> dict:
    normalized = normalize_columns(df)
    year_col = normalized.get("year") or normalized.get("ano")
    line_col = normalized.get("line") or normalized.get("conta") or normalized.get("linha")
    value_col = normalized.get("value") or normalized.get("valor") or normalized.get("total")
    date_col = normalized.get("date") or normalized.get("data")
    return {
        "year_col": year_col,
        "line_col": line_col,
        "value_col": value_col,
        "date_col": date_col,
    }


def coerce_year_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed.dt.year
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_base(df: pd.DataFrame, year_col: str, line_col: str, value_col: str) -> pd.Series:
    working = df.copy()
    working["__year"] = coerce_year_series(working[year_col])
    if working["__year"].isna().all():
        raise ValueError("Não foi possível interpretar o ano da base.")

    base_rows = working[working["__year"] == BASE_YEAR]
    if base_rows.empty:
        raise ValueError("A base precisa conter o ano de 2026.")

    grouped = base_rows.groupby(line_col, dropna=True)[value_col].sum()
    grouped.index = grouped.index.astype(str)
    return grouped


def build_blueprint() -> list[dict]:
    layout = []
    for group in GROUP_DEFINITIONS:
        layout.append({"type": "group", "name": group["name"], "children": group["children"]})
        for calc in CALC_LINES:
            if calc["formula"][0] == group["name"]:
                layout.append({"type": "calc", "name": calc["name"], "formula": calc["formula"]})
    return layout


def uniquify_children(children: list[str]) -> list[dict]:
    seen = {}
    items = []
    for child in children:
        key = child
        if key in seen:
            seen[key] += 1
            key = f"{child} ({seen[child]})"
        else:
            seen[key] = 1
        items.append({"key": key, "label": child})
    return items


def build_leaf_items() -> tuple[list[dict], dict]:
    leaf_items = []
    group_children = {}
    for group in GROUP_DEFINITIONS:
        children = uniquify_children(group["children"])
        group_children[group["name"]] = children
        for child in children:
            leaf_items.append({
                "group": group["name"],
                "key": child["key"],
                "label": child["label"],
            })
    return leaf_items, group_children


def build_base_input_items(leaf_items: list[dict]) -> list[dict]:
    items = []
    items.extend(leaf_items)
    for group in GROUP_DEFINITIONS:
        if not group["children"]:
            items.append({
                "group": group["name"],
                "key": group["name"],
                "label": group["name"],
            })
    return items


def initialize_base_values(items: list[dict]) -> pd.Series:
    index = [item["key"] for item in items]
    return pd.Series(0.0, index=index, dtype=float)


def compute_leaf_projection(base_values: pd.Series, leaf_items: list[dict], rates: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(index=[item["key"] for item in leaf_items], columns=ALL_YEARS, dtype=float)
    for item in leaf_items:
        label = item["label"]
        base_value = float(base_values.get(label, 0.0))
        data.loc[item["key"], BASE_YEAR] = base_value
        prev = base_value
        for year in PROJECTION_YEARS:
            pct = rates.loc[item["key"], year]
            prev = prev * (1 + pct / 100.0)
            data.loc[item["key"], year] = prev
    return data


def aggregate_groups(
    leaf_projection: pd.DataFrame,
    group_children: dict,
    base_values: pd.Series,
) -> dict:
    group_totals = {}
    for group, children in group_children.items():
        if not children:
            base_value = float(base_values.get(group, 0.0))
            series = pd.Series({year: base_value for year in ALL_YEARS})
            group_totals[group] = series
            continue
        keys = [child["key"] for child in children]
        group_totals[group] = leaf_projection.loc[keys].sum()
    return group_totals


def compute_calc_lines(group_totals: dict, calc_lines: list[dict]) -> dict:
    calc_totals = {}
    combined_totals = dict(group_totals)
    for calc in calc_lines:
        series = sum(combined_totals[name] for name in calc["formula"])
        calc_totals[calc["name"]] = series
        combined_totals[calc["name"]] = series
    return calc_totals


def build_final_table(
    layout: list[dict],
    group_totals: dict,
    leaf_projection: pd.DataFrame,
    group_children: dict,
    calc_totals: dict,
    leaf_items: list[dict],
) -> tuple[pd.DataFrame, set[str]]:
    rows = []
    calc_labels = set()
    leaf_label_map = {item["key"]: item["label"] for item in leaf_items}

    for item in layout:
        if item["type"] == "group":
            group_name = item["name"]
            group_series = group_totals[group_name]
            rows.append({"Label": group_name, **group_series.to_dict()})
            children = group_children[group_name]
            for child in children:
                child_series = leaf_projection.loc[child["key"]]
                rows.append({
                    "Label": f"    {leaf_label_map[child['key']]}",
                    **child_series.to_dict(),
                })
        elif item["type"] == "calc":
            calc_series = calc_totals[item["name"]]
            calc_labels.add(item["name"])
            rows.append({"Label": item["name"], **calc_series.to_dict()})

    final_df = pd.DataFrame(rows)
    return final_df, calc_labels


def format_pt_br(value: float) -> str:
    if pd.isna(value):
        return ""
    formatted = f"{abs(value):,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    if value < 0:
        return f"({formatted})"
    return formatted


def main() -> None:
    st.set_page_config(page_title="Projeção Anual 2026-2036", layout="wide")
    st.title("Projeção Anual (2026-2036)")

    st.sidebar.header("Base 2026")
    input_mode = st.sidebar.radio("Forma de input", options=["Manual", "Excel"], horizontal=True)
    uploaded_file = None
    if input_mode == "Excel":
        uploaded_file = st.sidebar.file_uploader("Carregar Excel (.xlsx)", type=["xlsx"])

    base_values = pd.Series(dtype=float)
    column_selection = None

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        defaults = detect_default_columns(df)

        with st.sidebar.expander("Mapeamento de colunas", expanded=False):
            columns = df.columns.tolist()
            year_col = st.selectbox("Coluna de Ano", options=columns, index=columns.index(defaults["year_col"]) if defaults["year_col"] in columns else 0)
            line_col = st.selectbox("Coluna de Linha", options=columns, index=columns.index(defaults["line_col"]) if defaults["line_col"] in columns else 0)
            value_col = st.selectbox("Coluna de Valor", options=columns, index=columns.index(defaults["value_col"]) if defaults["value_col"] in columns else 0)
            column_selection = {"year_col": year_col, "line_col": line_col, "value_col": value_col}

        if column_selection:
            try:
                base_values = load_base(df, **column_selection)
            except ValueError as exc:
                st.sidebar.error(str(exc))

    leaf_items, group_children = build_leaf_items()
    base_input_items = build_base_input_items(leaf_items)

    if "base_values" not in st.session_state:
        st.session_state.base_values = initialize_base_values(base_input_items)

    if not base_values.empty:
        updated_base = st.session_state.base_values.copy()
        for item in base_input_items:
            updated_base[item["key"]] = float(base_values.get(item["label"], updated_base[item["key"]]))
        st.session_state.base_values = updated_base

    if "rates" not in st.session_state:
        st.session_state.rates = pd.DataFrame(
            0.0,
            index=[item["key"] for item in leaf_items],
            columns=PROJECTION_YEARS,
        )

    st.sidebar.caption("A base 2026 é obrigatória para projetar 2027-2036.")

    tabs = st.tabs(["Base 2026", "Percentuais", "P&L Projetado"])

    with tabs[0]:
        st.subheader("Base 2026 (editável)")
        base_df = pd.DataFrame({
            "Linha": [item["label"] for item in base_input_items],
            "Valor 2026": [st.session_state.base_values.get(item["key"], 0.0) for item in base_input_items],
        })
        base_editor = st.data_editor(
            base_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={"Linha": st.column_config.TextColumn(disabled=True)},
        )
        if st.button("Atualizar base 2026"):
            updated = pd.Series(
                base_editor["Valor 2026"].astype(float).values,
                index=[item["key"] for item in base_input_items],
            )
            st.session_state.base_values = updated
            st.success("Base 2026 atualizada.")

    with tabs[1]:
        st.subheader("Percentuais de Projeção (Nível 2)")
        rates_df = st.session_state.rates.copy()
        rates_df.insert(0, "Linha", [item["label"] for item in leaf_items])

        editor = st.data_editor(
            rates_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={"Linha": st.column_config.TextColumn(disabled=True)},
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Recalcular projeção"):
                st.session_state.rates = editor.drop(columns=["Linha"]).astype(float)
                st.success("Projeção recalculada.")
        with col2:
            if st.button("Limpar percentuais"):
                st.session_state.rates = pd.DataFrame(
                    0.0,
                    index=st.session_state.rates.index,
                    columns=st.session_state.rates.columns,
                )
                st.success("Percentuais zerados.")

    with tabs[2]:
        st.subheader("P&L Projetado")
        base_values = st.session_state.base_values
        leaf_projection = compute_leaf_projection(base_values, leaf_items, st.session_state.rates)
        group_totals = aggregate_groups(leaf_projection, group_children, base_values)
        calc_totals = compute_calc_lines(group_totals, CALC_LINES)
        blueprint = build_blueprint()
        final_df, calc_labels = build_final_table(
            blueprint,
            group_totals,
            leaf_projection,
            group_children,
            calc_totals,
            leaf_items,
        )

        def style_rows(row):
            label = row["Label"].strip()
            if label in calc_labels:
                return ["font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled = final_df.style.format({year: format_pt_br for year in ALL_YEARS}).apply(style_rows, axis=1)
        st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
