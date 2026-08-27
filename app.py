import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import bisect

# ============================================================
# REE–Na–SO4 THERMODYNAMIC SCREENING SIMULATOR
# ============================================================

st.set_page_config(
    page_title="REE–Na–SO₄ Thermodynamic Screening",
    page_icon="🧪",
    layout="wide",
)

# ============================================================
# DATABASE
# ============================================================
# ATENÇÃO:
# Os Ksp abaixo são valores provisórios para desenvolvimento
# da interface. Devem ser substituídos por valores validados
# experimentalmente/literariamente antes de uso científico.
#
# Sistema:
# REE3+ + Na+ + 2 SO4(2-) ⇌ NaREE(SO4)2(s)
#
# Ksp = [Na+][REE3+][SO4(2-)]²
# ============================================================

REE_DATABASE = {
    "Sc": {
        "name": "Scandium",
        "ksp": 2.0e-5,
        "ksp_status": "placeholder",
    },
    "Y": {
        "name": "Yttrium",
        "ksp": 3.0e-5,
        "ksp_status": "placeholder",
    },
    "La": {
        "name": "Lanthanum",
        "ksp": 5.0e-5,
        "ksp_status": "placeholder",
    },
    "Ce": {
        "name": "Cerium",
        "ksp": 4.0e-5,
        "ksp_status": "placeholder",
    },
    "Pr": {
        "name": "Praseodymium",
        "ksp": 3.0e-5,
        "ksp_status": "placeholder",
    },
    "Nd": {
        "name": "Neodymium",
        "ksp": 1.0e-5,
        "ksp_status": "placeholder",
    },
    "Pm": {
        "name": "Promethium",
        "ksp": 1.5e-5,
        "ksp_status": "placeholder",
    },
    "Sm": {
        "name": "Samarium",
        "ksp": 8.0e-6,
        "ksp_status": "placeholder",
    },
    "Eu": {
        "name": "Europium",
        "ksp": 7.0e-6,
        "ksp_status": "placeholder",
    },
    "Gd": {
        "name": "Gadolinium",
        "ksp": 6.0e-6,
        "ksp_status": "placeholder",
    },
    "Tb": {
        "name": "Terbium",
        "ksp": 5.0e-6,
        "ksp_status": "placeholder",
    },
    "Dy": {
        "name": "Dysprosium",
        "ksp": 4.0e-6,
        "ksp_status": "placeholder",
    },
    "Ho": {
        "name": "Holmium",
        "ksp": 3.5e-6,
        "ksp_status": "placeholder",
    },
    "Er": {
        "name": "Erbium",
        "ksp": 3.0e-6,
        "ksp_status": "placeholder",
    },
    "Tm": {
        "name": "Thulium",
        "ksp": 2.5e-6,
        "ksp_status": "placeholder",
    },
    "Yb": {
        "name": "Ytterbium",
        "ksp": 2.0e-6,
        "ksp_status": "placeholder",
    },
    "Lu": {
        "name": "Lutetium",
        "ksp": 1.8e-6,
        "ksp_status": "placeholder",
    },
}

# ============================================================
# EDTA DATABASE
# ============================================================

EDTA_DATABASE = {
    "name": "EDTA",

    # Simplified H4Y protonation model
    "pKas": [
        2.00,
        2.67,
        6.16,
        10.26,
    ],

    # REE-EDTA assumed 1:1 complexes
    "coordination_number": 1,

    # Starter values for development.
    # Replace with curated literature values later.
    "log_beta": {
        "Sc": 23.0,
        "Y": 18.1,
        "La": 15.5,
        "Ce": 16.0,
        "Pr": 16.4,
        "Nd": 16.5,
        "Pm": 16.7,
        "Sm": 17.0,
        "Eu": 17.1,
        "Gd": 17.4,
        "Tb": 17.7,
        "Dy": 18.0,
        "Ho": 18.2,
        "Er": 18.4,
        "Tm": 18.6,
        "Yb": 18.7,
        "Lu": 18.9,
    },
}

# Sulfate protonation
SULFATE_PKA = 1.99


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_inverse_alpha(ph_value, pkas_list):
    """
    Calculates 1/alpha for protonation side reactions.
    """

    if not pkas_list:
        return 1.0

    free_h = 10.0 ** (-ph_value)

    ka_constants = [
        10.0 ** (-pka)
        for pka in pkas_list
    ]

    alpha_sum = 1.0
    cumulative_ka_product = 1.0

    for i, ka in enumerate(ka_constants[::-1]):

        cumulative_ka_product *= ka

        alpha_sum += (
            free_h ** (i + 1)
        ) / cumulative_ka_product

    return alpha_sum


def solve_fully_coupled_system(
    metal_systems,
    total_precipitant,
    total_cation_2,
    inv_alpha_A,
    inv_alpha_Y,
    total_chelator_concentration,
    is_active,
):
    """
    Solves the coupled Na+/SO4/REE/EDTA equilibrium
    using nested bisection loops.
    """

    def precipitant_residual_loop(test_free_precipitant):

        test_free_precipitant = max(
            test_free_precipitant,
            1e-45,
        )

        active_anion = (
            test_free_precipitant
            / inv_alpha_A
        )

        active_anion_for_ksp = max(
            1e-45,
            active_anion,
        )

        # ----------------------------------------------------
        # Solve free Na+
        # ----------------------------------------------------

        low_na = 0.0
        high_na = max(
            total_cation_2,
            1e-30,
        )

        for _ in range(40):

            mid_na = (
                low_na + high_na
            ) / 2.0

            # ------------------------------------------------
            # Solve free EDTA
            # ------------------------------------------------

            if (
                is_active
                and total_chelator_concentration > 0.0
            ):

                low_y = 0.0

                high_y = max(
                    total_chelator_concentration
                    / inv_alpha_Y,
                    1e-30,
                )

                for _ in range(35):

                    mid_y = (
                        low_y + high_y
                    ) / 2.0

                    calc_chelator = (
                        mid_y * inv_alpha_Y
                    )

                    for metal in metal_systems:

                        denominator = (
                            max(
                                1e-45,
                                mid_na,
                            )
                            * active_anion_for_ksp ** 2
                        )

                        max_free_m_ksp = (
                            metal["ksp"]
                            / denominator
                        )

                        beta_term = (
                            metal["beta_stability"]
                            * (
                                mid_y
                                ** metal["coord_number_N"]
                            )
                        )

                        max_total_soluble_m = (
                            max_free_m_ksp
                            * (
                                1.0
                                + beta_term
                            )
                        )

                        real_total_soluble_m = min(
                            metal["initial_conc"],
                            max_total_soluble_m,
                        )

                        real_free_metal = (
                            real_total_soluble_m
                            / (
                                1.0
                                + beta_term
                            )
                        )

                        calc_chelator += (
                            metal["coord_number_N"]
                            * metal["beta_stability"]
                            * real_free_metal
                            * (
                                mid_y
                                ** metal["coord_number_N"]
                            )
                        )

                    if (
                        calc_chelator
                        < total_chelator_concentration
                    ):
                        low_y = mid_y
                    else:
                        high_y = mid_y

                stable_free_y = (
                    low_y + high_y
                ) / 2.0

            else:

                stable_free_y = 0.0

            # ------------------------------------------------
            # Sodium mass balance
            # ------------------------------------------------

            calculated_na = mid_na

            for metal in metal_systems:

                denominator = (
                    max(
                        1e-45,
                        mid_na,
                    )
                    * active_anion_for_ksp ** 2
                )

                max_free_m_ksp = (
                    metal["ksp"]
                    / denominator
                )

                if stable_free_y > 0:

                    beta_term = (
                        metal["beta_stability"]
                        * (
                            stable_free_y
                            ** metal["coord_number_N"]
                        )
                    )

                else:

                    beta_term = 0.0

                max_total_soluble_m = (
                    max_free_m_ksp
                    * (
                        1.0
                        + beta_term
                    )
                )

                real_total_soluble_m = min(
                    metal["initial_conc"],
                    max_total_soluble_m,
                )

                precipitated_m = (
                    metal["initial_conc"]
                    - real_total_soluble_m
                )

                calculated_na += (
                    precipitated_m
                )

            if calculated_na < total_cation_2:

                low_na = mid_na

            else:

                high_na = mid_na

        stable_na = (
            low_na + high_na
        ) / 2.0

        # ----------------------------------------------------
        # Sulfate mass balance
        # ----------------------------------------------------

        calculated_total_precipitant = (
            test_free_precipitant
            * inv_alpha_A
        )

        for metal in metal_systems:

            denominator = (
                max(
                    1e-45,
                    stable_na,
                )
                * active_anion_for_ksp ** 2
            )

            max_free_m_ksp = (
                metal["ksp"]
                / denominator
            )

            if stable_free_y > 0:

                beta_term = (
                    metal["beta_stability"]
                    * (
                        stable_free_y
                        ** metal["coord_number_N"]
                    )
                )

            else:

                beta_term = 0.0

            max_total_soluble_m = (
                max_free_m_ksp
                * (
                    1.0
                    + beta_term
                )
            )

            real_total_soluble_m = min(
                metal["initial_conc"],
                max_total_soluble_m,
            )

            precipitated_m = (
                metal["initial_conc"]
                - real_total_soluble_m
            )

            calculated_total_precipitant += (
                2.0
                * precipitated_m
            )

        return (
            calculated_total_precipitant
            - total_precipitant
        )

    # ========================================================
    # Solve free sulfate
    # ========================================================

    try:

        upper_bound = max(
            total_precipitant + 0.1,
            1e-8,
        )

        if (
            precipitant_residual_loop(0.0)
            > 0
        ):

            solved_free_precipitant = 0.0

        else:

            solved_free_precipitant = bisect(
                precipitant_residual_loop,
                0.0,
                upper_bound,
                xtol=1e-12,
            )

    except Exception:

        solved_free_precipitant = 0.0

    # ========================================================
    # Final state
    # ========================================================

    active_anion = (
        solved_free_precipitant
        / inv_alpha_A
    )

    active_anion_for_ksp = max(
        1e-45,
        active_anion,
    )

    low_na = 0.0
    high_na = max(
        total_cation_2,
        1e-30,
    )

    for _ in range(40):

        mid_na = (
            low_na + high_na
        ) / 2.0

        # ----------------------------------------------------
        # EDTA equilibrium
        # ----------------------------------------------------

        if (
            is_active
            and total_chelator_concentration > 0.0
        ):

            low_y = 0.0

            high_y = max(
                total_chelator_concentration
                / inv_alpha_Y,
                1e-30,
            )

            for _ in range(35):

                mid_y = (
                    low_y + high_y
                ) / 2.0

                calc_chelator = (
                    mid_y * inv_alpha_Y
                )

                for metal in metal_systems:

                    denominator = (
                        max(
                            1e-45,
                            mid_na,
                        )
                        * active_anion_for_ksp ** 2
                    )

                    max_free_m_ksp = (
                        metal["ksp"]
                        / denominator
                    )

                    beta_term = (
                        metal["beta_stability"]
                        * (
                            mid_y
                            ** metal["coord_number_N"]
                        )
                    )

                    max_total_soluble_m = (
                        max_free_m_ksp
                        * (
                            1.0
                            + beta_term
                        )
                    )

                    real_total_soluble_m = min(
                        metal["initial_conc"],
                        max_total_soluble_m,
                    )

                    real_free_metal = (
                        real_total_soluble_m
                        / (
                            1.0
                            + beta_term
                        )
                    )

                    calc_chelator += (
                        metal["coord_number_N"]
                        * metal["beta_stability"]
                        * real_free_metal
                        * (
                            mid_y
                            ** metal["coord_number_N"]
                        )
                    )

                if (
                    calc_chelator
                    < total_chelator_concentration
                ):

                    low_y = mid_y

                else:

                    high_y = mid_y

            stable_free_y = (
                low_y + high_y
            ) / 2.0

        else:

            stable_free_y = 0.0

        # ----------------------------------------------------
        # Sodium mass balance
        # ----------------------------------------------------

        calculated_na = mid_na

        for metal in metal_systems:

            denominator = (
                max(
                    1e-45,
                    mid_na,
                )
                * active_anion_for_ksp ** 2
            )

            max_free_m_ksp = (
                metal["ksp"]
                / denominator
            )

            if stable_free_y > 0:

                beta_term = (
                    metal["beta_stability"]
                    * (
                        stable_free_y
                        ** metal["coord_number_N"]
                    )
                )

            else:

                beta_term = 0.0

            max_total_soluble_m = (
                max_free_m_ksp
                * (
                    1.0
                    + beta_term
                )
            )

            real_total_soluble_m = min(
                metal["initial_conc"],
                max_total_soluble_m,
            )

            precipitated_m = (
                metal["initial_conc"]
                - real_total_soluble_m
            )

            calculated_na += (
                precipitated_m
            )

        if calculated_na < total_cation_2:

            low_na = mid_na

        else:

            high_na = mid_na

    final_free_na = (
        low_na + high_na
    ) / 2.0

    # ========================================================
    # Precipitation efficiencies
    # ========================================================

    efficiencies_row = {}

    for metal in metal_systems:

        denominator = (
            max(
                1e-45,
                final_free_na,
            )
            * active_anion_for_ksp ** 2
        )

        max_free_m_ksp = (
            metal["ksp"]
            / denominator
        )

        if stable_free_y > 0:

            beta_term = (
                metal["beta_stability"]
                * (
                    stable_free_y
                    ** metal["coord_number_N"]
                )
            )

        else:

            beta_term = 0.0

        max_total_soluble_m = (
            max_free_m_ksp
            * (
                1.0
                + beta_term
            )
        )

        real_total_soluble_m = min(
            metal["initial_conc"],
            max_total_soluble_m,
        )

        percentage = (
            (
                metal["initial_conc"]
                - real_total_soluble_m
            )
            / metal["initial_conc"]
            * 100.0
        )

        efficiencies_row[
            metal["name"]
        ] = float(
            np.clip(
                percentage,
                0,
                100,
            )
        )

    return efficiencies_row


# ============================================================
# SIMULATION
# ============================================================

@st.cache_data(show_spinner=False)
def run_simulation(
    selected_symbols,
    concentrations,
    total_chelator_concentration,
    use_chelating_agent,
    fixed_ph_value,
    fixed_sodium_conc,
    fixed_sulfate_conc,
    ph_start,
    ph_end,
    max_sodium_addition,
    max_sulfate_addition,
    n_points=120,
):

    metal_systems = []

    for symbol, concentration in zip(
        selected_symbols,
        concentrations,
    ):

        database_entry = (
            REE_DATABASE[symbol]
        )

        metal_systems.append({

            "name": symbol,

            "initial_conc": float(
                concentration
            ),

            "ksp": database_entry["ksp"],

            "coord_number_N": (
                EDTA_DATABASE[
                    "coordination_number"
                ]
            ),

            "beta_stability": (
                10.0
                ** EDTA_DATABASE[
                    "log_beta"
                ][symbol]
            ),
        })

    is_active = (
        use_chelating_agent
        and total_chelator_concentration > 0
    )

    # ========================================================
    # MODE 1 — pH
    # ========================================================

    ph_range = np.linspace(
        ph_start,
        ph_end,
        n_points,
    )

    mode_1 = []

    for ph in ph_range:

        inv_A = calculate_inverse_alpha(
            ph,
            [SULFATE_PKA],
        )

        inv_Y = calculate_inverse_alpha(
            ph,
            EDTA_DATABASE["pKas"],
        )

        row = solve_fully_coupled_system(
            metal_systems,
            fixed_sulfate_conc,
            fixed_sodium_conc,
            inv_A,
            inv_Y,
            total_chelator_concentration,
            is_active,
        )

        row["pH"] = ph

        mode_1.append(row)

    # ========================================================
    # Fixed alpha values for modes 2 and 3
    # ========================================================

    inv_A_fixed = calculate_inverse_alpha(
        fixed_ph_value,
        [SULFATE_PKA],
    )

    inv_Y_fixed = calculate_inverse_alpha(
        fixed_ph_value,
        EDTA_DATABASE["pKas"],
    )

    # ========================================================
    # MODE 2 — Na+
    # ========================================================

    sodium_range = np.linspace(
        0,
        max_sodium_addition,
        n_points,
    )

    mode_2 = []

    for sodium in sodium_range:

        row = solve_fully_coupled_system(
            metal_systems,
            fixed_sulfate_conc,
            sodium,
            inv_A_fixed,
            inv_Y_fixed,
            total_chelator_concentration,
            is_active,
        )

        row["cation_conc"] = sodium

        mode_2.append(row)

    # ========================================================
    # MODE 3 — Sulfate
    # ========================================================

    sulfate_range = np.linspace(
        0,
        max_sulfate_addition,
        n_points,
    )

    mode_3 = []

    for sulfate in sulfate_range:

        row = solve_fully_coupled_system(
            metal_systems,
            sulfate,
            fixed_sodium_conc,
            inv_A_fixed,
            inv_Y_fixed,
            total_chelator_concentration,
            is_active,
        )

        row["precipitant_conc"] = sulfate

        mode_3.append(row)

    return (
        pd.DataFrame(mode_1),
        pd.DataFrame(mode_2),
        pd.DataFrame(mode_3),
    )


# ============================================================
# PLOT
# ============================================================

def make_plot(
    dataframe,
    x_column,
    metals,
    title,
    x_label,
):

    figure = go.Figure()

    for metal in metals:

        figure.add_trace(
            go.Scatter(
                x=dataframe[x_column],
                y=dataframe[metal],
                mode="lines+markers",
                name=metal,
                marker={
                    "size": 5
                },
            )
        )

    figure.update_layout(

        title=title,

        xaxis_title=x_label,

        yaxis_title=(
            "% precipitated "
            "as double salt"
        ),

        yaxis={
            "range": [
                -2,
                102,
            ]
        },

        hovermode="x unified",

        height=480,

        legend_title="REE",

        margin={
            "l": 50,
            "r": 20,
            "t": 70,
            "b": 50,
        },
    )

    figure.update_xaxes(
        showgrid=True
    )

    figure.update_yaxes(
        showgrid=True
    )

    return figure


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🧪 REE–Na–SO₄ Thermodynamic Screening"
)

st.markdown(
    """
    ### Rare-earth / sodium sulfate double-salt simulator

    Select the rare-earth elements and provide their initial
    concentrations. Chemical constants such as **Ksp, EDTA
    pKa and EDTA stability constants are retrieved automatically
    from the internal database.**
    """
)

st.info(
    """
    ⚠️ **Development database:** the Ksp and EDTA constants
    currently embedded in this application are starter values
    for interface development. They must be replaced and
    validated against the literature before quantitative
    scientific use.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    selected_symbols = st.multiselect(

        "Rare-earth elements",

        options=list(
            REE_DATABASE.keys()
        ),

        default=[
            "Nd",
            "La",
            "Pr",
        ],

        format_func=lambda symbol:
            (
                f"{symbol} — "
                f"{REE_DATABASE[symbol]['name']}"
            ),
    )

    if not selected_symbols:

        st.warning(
            "Select at least one REE."
        )

        st.stop()

    # --------------------------------------------------------
    # Concentrations
    # --------------------------------------------------------

    st.subheader(
        "Initial REE concentrations"
    )

    concentrations = []

    for symbol in selected_symbols:

        default_value = (
            0.05
            if symbol in [
                "Nd",
                "La",
            ]
            else 0.02
        )

        concentrations.append(

            st.number_input(

                f"[{symbol}]₀ (mol/L)",

                min_value=1e-8,

                value=default_value,

                step=0.005,

                format="%.5f",

                key=f"conc_{symbol}",
            )
        )

    # --------------------------------------------------------
    # EDTA
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧬 Complexation"
    )

    use_chelating_agent = st.toggle(
        "Use EDTA",
        value=True,
    )

    if use_chelating_agent:

        total_chelator_concentration = (
            st.number_input(
                "Total EDTA (mol/L)",
                min_value=0.0,
                value=0.30,
                step=0.01,
                format="%.3f",
            )
        )

        st.caption(
            "EDTA pKa: "
            + ", ".join(
                map(
                    str,
                    EDTA_DATABASE["pKas"],
                )
            )
        )

    else:

        total_chelator_concentration = 0.0

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Baseline conditions"
    )

    fixed_ph_value = st.number_input(
        "Fixed pH for scans 2–3",
        min_value=0.0,
        max_value=14.0,
        value=3.0,
        step=0.1,
    )

    fixed_sodium_conc = st.number_input(
        "Total Na⁺ (mol/L)",
        min_value=0.0,
        value=0.20,
        step=0.01,
        format="%.3f",
    )

    fixed_sulfate_conc = st.number_input(
        "Total sulfate (mol/L)",
        min_value=0.0,
        value=0.50,
        step=0.01,
        format="%.3f",
    )

    # --------------------------------------------------------
    # Scan configuration
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Scan ranges"
    )

    ph_start, ph_end = st.slider(
        "pH range",
        min_value=0.0,
        max_value=14.0,
        value=(0.0, 7.0),
        step=0.1,
    )

    max_sodium_addition = st.number_input(
        "Maximum Na⁺ (mol/L)",
        min_value=0.0,
        value=1.0,
        step=0.05,
        format="%.3f",
    )

    max_sulfate_addition = st.number_input(
        "Maximum sulfate (mol/L)",
        min_value=0.0,
        value=1.0,
        step=0.05,
        format="%.3f",
    )

    st.divider()

    run = st.button(
        "▶ Run simulation",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RUN
# ============================================================

if "results" not in st.session_state:

    st.session_state.results = None


if run:

    with st.spinner(
        "Solving coupled thermodynamic system…"
    ):

        st.session_state.results = (
            run_simulation(

                tuple(
                    selected_symbols
                ),

                tuple(
                    concentrations
                ),

                total_chelator_concentration,

                use_chelating_agent,

                fixed_ph_value,

                fixed_sodium_conc,

                fixed_sulfate_conc,

                ph_start,

                ph_end,

                max_sodium_addition,

                max_sulfate_addition,
            )
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

if st.session_state.results is None:

    st.markdown(
        """
        ## 👈 Configure the system

        Select your REEs, enter their initial concentrations,
        choose whether to use EDTA, and click **Run simulation**.

        The simulator will calculate:

        **1. Separation vs pH**

        **2. Separation vs Na⁺ dosage**

        **3. Separation vs sulfate dosage**
        """
    )

    with st.expander(
        "📚 View embedded database"
    ):

        database_rows = []

        for symbol, entry in (
            REE_DATABASE.items()
        ):

            database_rows.append({

                "REE": symbol,

                "Name": entry["name"],

                "Ksp":
                    f"{entry['ksp']:.2e}",

                "Ksp status":
                    entry["ksp_status"],

                "EDTA log β":
                    EDTA_DATABASE[
                        "log_beta"
                    ][symbol],
            })

        st.dataframe(
            pd.DataFrame(
                database_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

df_mode_1, df_mode_2, df_mode_3 = (
    st.session_state.results
)

st.success(
    "Simulation completed successfully."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Separation vs pH",
        "🧂 Separation vs Na⁺",
        "🧪 Separation vs sulfate",
        "📚 Database",
    ]
)


chelating_label = (

    f"with {total_chelator_concentration:g} M EDTA"

    if (
        use_chelating_agent
        and total_chelator_concentration > 0
    )

    else
    "without EDTA"
)


# ============================================================
# TAB 1 — pH
# ============================================================

with tab1:

    st.plotly_chart(

        make_plot(

            df_mode_1,

            "pH",

            selected_symbols,

            (
                "REE precipitation efficiency "
                f"vs pH — {chelating_label}"
            ),

            "pH",
        ),

        use_container_width=True,
    )

    st.dataframe(
        df_mode_1,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 2 — Na+
# ============================================================

with tab2:

    st.plotly_chart(

        make_plot(

            df_mode_2,

            "cation_conc",

            selected_symbols,

            (
                "REE precipitation efficiency "
                f"vs Na⁺ dosage — pH {fixed_ph_value:g}"
            ),

            "Total Na⁺ (mol/L)",
        ),

        use_container_width=True,
    )

    st.dataframe(
        df_mode_2,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3 — Sulfate
# ============================================================

with tab3:

    st.plotly_chart(

        make_plot(

            df_mode_3,

            "precipitant_conc",

            selected_symbols,

            (
                "REE precipitation efficiency "
                f"vs sulfate dosage — pH {fixed_ph_value:g}"
            ),

            "Total sulfate (mol/L)",
        ),

        use_container_width=True,
    )

    st.dataframe(
        df_mode_3,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 4 — DATABASE
# ============================================================

with tab4:

    st.subheader(
        "📚 Embedded chemical database"
    )

    database_rows = []

    for symbol in selected_symbols:

        entry = REE_DATABASE[symbol]

        database_rows.append({

            "REE":
                symbol,

            "Name":
                entry["name"],

            "Ksp":
                entry["ksp"],

            "Ksp status":
                entry["ksp_status"],

            "EDTA pKa":
                ", ".join(
                    map(
                        str,
                        EDTA_DATABASE["pKas"],
                    )
                ),

            "EDTA log β":
                EDTA_DATABASE[
                    "log_beta"
                ][symbol],

            "EDTA N":
                EDTA_DATABASE[
                    "coordination_number"
                ],
        })

    st.dataframe(

        pd.DataFrame(
            database_rows
        ),

        use_container_width=True,

        hide_index=True,
    )

    st.caption(
        "Sulfate protonation model: pKa₂ = 1.99."
    )


# ============================================================
# EXPORT
# ============================================================

st.divider()

st.subheader(
    "📥 Export results"
)

combined_results = []

for scan_name, dataframe in [

    (
        "pH scan",
        df_mode_1,
    ),

    (
        "Na+ scan",
        df_mode_2,
    ),

    (
        "Sulfate scan",
        df_mode_3,
    ),
]:

    temporary_dataframe = dataframe.copy()

    temporary_dataframe.insert(
        0,
        "scan",
        scan_name,
    )

    combined_results.append(
        temporary_dataframe
    )


download_dataframe = pd.concat(
    combined_results,
    ignore_index=True,
)


st.download_button(

    "Download results as CSV",

    data=(
        download_dataframe
        .to_csv(index=False)
        .encode("utf-8")
    ),

    file_name=(
        "REE_Na_SO4_simulation_results.csv"
    ),

    mime="text/csv",
)
