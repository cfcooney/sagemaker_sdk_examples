def summarize_hpo_results(tuning_job_name):
    """
    Query tuning results and display the best score,
    parameters, and job-name
    
    Parameters:
        tuning_job_name: str
            name of tuning passed to sagemaker
            
    Return:
        hpo_results: cudf.DataFrame
            Results of all HPO iterations
    """
    
    hpo_results = boto3.Session().client(
        'sagemaker'
        ).describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )

    best_job = hpo_results['BestTrainingJob']['TrainingJobName']
    best_score = hpo_results['BestTrainingJob']['FinalHyperParameterTuningJobObjectiveMetric']['Value']  # noqa
    best_params = hpo_results['BestTrainingJob']['TunedHyperParameters']
    print(f'best score: {best_score}')
    print(f'best params: {best_params}')
    print(f'best job-name: {best_job}')