
# survival analysis
def run_survival_analysis(final_patient):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    print(final_patient["duration"].dtype)
    print(final_patient["duration"].head())
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))

    for group in ['asthma', 'copd']:
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

    result = logrank_test(
        asthma['duration'],
        copd['duration'],
        event_observed_A=asthma['event'],
        event_observed_B=copd['event']
    )

    print(result.summary)