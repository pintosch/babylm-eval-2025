import mlflow

update_dict = {
    "blimp_filtered": 75.56,
    "blimp_supplement": 61.98,
    "comps": 56.2,
    "entity_tracking": 18.96,
    "ewok": 52.09,
    "wug_adj": 0.54,
    "wug_past": 0.03,
    "vqa": 33.2,
    "winoground": 49.6,
}


def main():
    run_id = "7e8e4787bccf4c0aaed4513f66525ee7"
    with mlflow.start_run(run_id=run_id):
        for key, value in update_dict.items():
            mlflow.log_metric(key, value)
    mlflow.end_run()


if __name__ == "__main__":
    main()
