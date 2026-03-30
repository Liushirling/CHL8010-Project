# survival analysis
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def KM_curves_and_LRT(final_patient):
    """
    Input: cleaned final patient data. 
        - Unique patient_ID
        - exposure (asthma / copd / aco)
        - event (1 if depression, else 0)
        - duration (time from index_date to event or censor)
    Output: 
        - KM curves plot, 
        - Pairwise Log-Rank test result (asthma/copd, copd/aco, asthma/aco) 
    """
    print(final_patient["duration"].dtype)
    print(final_patient["duration"].head())
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))

    for group in ['asthma', 'copd', 'aco']:
        mask = final_patient['exposure'] == group

        kmf.fit(
            durations=final_patient.loc[mask, 'duration'],
            event_observed=final_patient.loc[mask, 'event'],
            label=group.capitalize()
        )

        kmf.plot_survival_function(ci_show=True)

    plt.title('Kaplan-Meier Curve: Time to Depression')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

    # log-rank test
    asthma = final_patient[final_patient['exposure'] == 'asthma']
    copd   = final_patient[final_patient['exposure'] == 'copd']
    aco = final_patient[final_patient['exposure'] == 'aco']

    result_asthma_copd = logrank_test(
        asthma['duration'],
        copd['duration'],
        event_observed_A=asthma['event'],
        event_observed_B=copd['event']
    )

    result_copd_aco = logrank_test(
        copd['duration'],
        aco['duration'],
        event_observed_A=copd['event'],
        event_observed_B=aco['event']
    )

    result_asthma_aco = logrank_test(
        asthma['duration'],
        aco['duration'],
        event_observed_A=asthma['event'],
        event_observed_B=aco['event']
    )

    print(result_asthma_copd.summary, result_copd_aco.summary, result_asthma_aco.summary)

# Confounder analysis 
def confounder_table(df, duration_col, event_col, exposure_var, confounders):
    """
    Build a confounder analysis table.

    Parameters
    ----------
    df : DataFrame
    exposure_var : str (single exposure variable, e.g. "exposure_asthma")
    confounders : list of confounders

    Returns
    -------
    table : DataFrame
    summary_text : str
    """
    
    cph = CoxPHFitter()
    results = []

    # Step 1: crude model
    df_crude = df[[duration_col, event_col, exposure_var]].dropna()
    cph.fit(df_crude, duration_col=duration_col, event_col=event_col)
    crude_hr = cph.summary.loc[exposure_var, "exp(coef)"]

    # Step 2: one-by-one adjustment
    for conf in confounders:
        df_temp = df[[duration_col, event_col, exposure_var, conf]].dropna()

        cph.fit(df_temp, duration_col=duration_col, event_col=event_col)
        summary = cph.summary

        hr_adj = summary.loc[exposure_var, "exp(coef)"]
        p_val = summary.loc[conf, "p"]

        pct_change = (hr_adj - crude_hr) / crude_hr * 100

        results.append({
            "Confounder": conf,
            "HR_crude": crude_hr,
            "HR_adjusted": hr_adj,
            "Percent_change": pct_change,
            "p_value": p_val
        })

    table = pd.DataFrame(results)

    # rounding for presentation
    table["HR_crude"] = table["HR_crude"].round(3)
    table["HR_adjusted"] = table["HR_adjusted"].round(3)
    table["Percent_change"] = table["Percent_change"].round(2)
    table["p_value"] = table["p_value"].round(4)

    # Step 3: overall (fully adjusted)
    df_full = df[[duration_col, event_col, exposure_var] + confounders].dropna()
    cph.fit(df_full, duration_col=duration_col, event_col=event_col)

    hr_full = cph.summary.loc[exposure_var, "exp(coef)"]
    pct_full = (hr_full - crude_hr) / crude_hr * 100

    summary_text = (
        f"After adjusting for all confounders, the hazard ratio for {exposure_var} "
        f"changed from {crude_hr:.3f} to {hr_full:.3f} "
        f"({pct_full:.1f}% change)."
    )

    return table, summary_text

# Pipeline 
def run_survival_analysis(final_patient):
    # --- Load final paitnet data ---
    df = final_patient.copy()
    
    # --- KM curves and log rank test result ----
    KM_curves_and_LRT(df)

    # --- Confounder finding---
    # set the exposure variable to dummy variables 
    df = pd.get_dummies(df, columns=["exposure"], drop_first=True)

    # run confounder 
    confounders = ["age", "Sex", "Num_Comorbidities", "diabetes", "Family_History"]
    confounder_result_copd, confounder_summary_copd = confounder_table(
        df=df,
        duration_col="duration",
        event_col="event",
        exposure_var="exposure_copd",
        confounders=confounders
    )
    confounder_result_asthma, confounder_summary_asthma = confounder_table(
        df=df,
        duration_col="duration",
        event_col="event",
        exposure_var="exposure_asthma",
        confounders=confounders
    )
    print(confounder_result_copd)
    print("\n", confounder_summary_copd)
    print(confounder_result_asthma)
    print("\n", confounder_summary_asthma)


    print(result.summary)
