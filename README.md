# ML/DL Project Tracking App

This is a simple yet powerful and multifunctional Streamlit app that helps keep track of ML/DL projects from start to finish. Whether it's managing datasets, monitoring training, or running quick inferences.

## Why Use This?

This app saves time and makes it easier to track and manage ML projects without jumping between multiple tools. 
I had a problem of going here and there between separate tools, which caused a sense of unclear vision of a whole project and wanted to zoom out a little bit. That's why the idea of collecting everything related to a project into a single place came up. Moreover, having such tool would easily help project leaders to track the process and overall job done.

## ClearML?
Although it might seem to be the cheap imitation of CLearML, it is not. ClearMl provides vast functionality in data & model versioning, training and inferencing and etc., but still you would need the place to track the dataset creation/annotation like CVAT, the task tracking tool like Jira or Linear, separate data visualization tool or you might just want to develop a model first before uploading it to ClearMl and want to evaluate intermediary results. That's why this app might be handy. Also, it is still possible to integrate all APIs in this app, shortly, sky is the limit :)

## How to best use it?
First of all, manage all your tools used to track your projects separately (in case you don't use complex open-source or company subscribed tool): prepare dataset regulating and visualizing tools, model version controlling tool, files version controlling tool, task management tool. Prepare your training and evaluation scripts and create APIs that would accept files and folders to be trained on and inferenced.

## How to deploy it locally?
See [How to deploy locally](https://github.com/Abulegenov/Local_DEPLOY_with_SCREEN))


## Overall Structure
- **General Information**: Provides the general information about the project: models used, pipeline, expected results, tools integrated, might include team contact information. 
- **Dataset Information**: Provides the information about the current dataset amount & nature. Includes links to external dataset visualization tools like FiftyOne. Could have progress bar into achieving the dataset number
- **Add Dataset**: Uses the CVAT API to keep an eye on dataset annotations' statuses and assignees. Includes methods to download ready (Completed) annotations and add those annotation files into current annotation file. Also, you can include AWS S3 APIs or any cloud storage your company uses for dataset storage and run them whenever new dataset is created.
- **Train**: Lets you choose the dataset to run the training by requesting into running FastAPI backend of training script as well as indicating hyperparameters. Includes the link to MLflow for tracking the losses and etc (might as well be visualized in this tab by analyzing mlflow model running artefacts)
- **Task & Hypothesis & Updates Management**: Keeps track of project tasks, hypotheses, and updates. Provides the list of tasks from Linear or Jira to be completed within this week. Project leaders also might write their comments and developers might log their updates.
- **Inference on single image**: Upload an image (or datapoint) and run a quick inference on selected model.
- **Model Comparison**: Select two models to compare the results on single image (or datapoint).

## Important Note
Be aware that this is a vision and experience of a single ML/DL practicioner. If you have suggestions or find this tool insightful feel free to contact me anytime.
Use the script as a template and integrate it with your own system.

## Conclusion

This dashboard significantly improves **workflow efficiency** by consolidating various tools into a **single interface**. It reduces **context-switching**, ensures **real-time tracking**, and facilitates **seamless collaboration** between data scientists, engineers, and project managers. Whether you're managing datasets, running experiments, or tracking team progress, this tool enhances productivity and project visibility.
