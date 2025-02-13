import streamlit as st
import json
import pandas as pd
from datetime import date
import requests, os,shutil
import time
from cvat_sdk import models
from cvat_sdk.api_client import Configuration, ApiClient
from cvat_sdk.api_client.model.data_request import DataRequest
import zipfile
from coco_annotations_add import *
from datetime import date, timedelta, datetime
import cv2

configuration = Configuration(
    host = "your_cvat_url",
    username = 'username',
    password = 'password',
                            )
cvat_url =  'your_cvat_url'
cvat_user = 'username'
cvat_pwd = 'password'
session = requests.Session()
session.auth = (cvat_user,cvat_pwd)
linear_token = 'your_linear_token'
def clear_updates():
    st.session_state["update_creation"] = ""

with st.sidebar:
    project_choice = st.selectbox('Please choose your project', ['Project_1','Project_2', 'Project_3'])

tabs = st.tabs(['General Information', 'Dataset Information','Add Dataset',
'Train','Tasks&Hypothesis&Updates','Inference on Image',
'Check between models'])

with tabs[0]: #'General Information'
    """This tab would have the general information regarding the project"""
    st.header(project_choice)
    st.header('The pipeline of project is: ')
    st.write('You can write here your overview of projet pipeline')
    #Or you can illustrate it with image
    st.image('project_pipeline.jpg',caption = 'Project Pipeline')
    st.header('Project Objectives')
    st.write('You can write here your project objectives')
    st.header('Project Tools')
    st.write('You can list here your project tools for each step')
    st.image('project_tools.jpg', caption = 'Project Tools')
    st.header('Project Models')
    st.write('You can list here your project models')
    st.image('project_models.jpg', caption = 'Project Models')
    
with tabs[1]: #'Dataset Information'
    """This tab would describe your current dataset"""
    st.header('Current Dataset')
    #first upload your annotation files (in my case it is COCO annotation files) or just list the number of images for each of the classes
    f = open('model_1.json')
    json_1 = json.load(f)
    f = open('model_2.json')
    json_2 = json.load(f)
    f = open('model_3.json')
    json_3 = json.load(f)
    num_class_model_1 = json_1['images'][-1]['id']
    num_class_model_2 = json_2['images'][-1]['id']
    num_class_model_3 = json_3['images'][-1]['id']
    dict_dataset_current = {
                        'class_1':[num_class_model_1,None, None],
                        'class_2':[None,num_class_model_2, None],
                        'class_3':[None,None,num_class_model_3],
                        }
    st.bar_chart(pd.DataFrame.from_dict(dict_dataset_current, orient='index', columns=['class_1','class_2','class_3']))

    #If your team uses open-source project tracking tool, you can add a link to your project dataset cards:
    tool = 'LINEAR'
    st.header(f'Check {tool} Datasets and Statuses')
    url_tool= f'https://{tool.lower()}/datasets' #example url

    st.markdown(f'''
                <a href={url_tool}><button style="background-color:LightBlue; font-size: 42px">{tool}</button></a>
                ''', unsafe_allow_html=True)
    
    #If your team uses dataset visualization tools, like Voxel51, you could also add their link
    st.header('Open Voxel51 Tool')
    url_voxel = 'url/datasets/dataset_name'

    st.markdown(f'''
                <a href={url_voxel}><button style="background-color:LightBlue; font-size: 42px">FiftyOne Dataset</button></a>
                ''', unsafe_allow_html=True)
    
    #it is convenient to have a dataset amount goal and your progress towards that goal:
    #use progress bars
    deadline = 'end of year'
    st.header(f'Please Enter number of data for each class of Project you want to have by {deadline}')
    segment_goal = st.text_input('', key = 'goal_set', value = '6000')
    
    st.header(f'Goal of {segment_goal} images is achieved by: ')
    st.write('Model 1: ', f'{round(num_class_model_1/int(segment_goal)*100)}%')
    bar_1 = st.progress(int(num_class_model_1/int(segment_goal)*100))
    
    st.write('Model 2: ', f'{round(num_class_model_2/int(segment_goal)*100)}%')
    bar_2 = st.progress(int(num_class_model_2/int(segment_goal)*100))

    st.write('Model 3: ', f'{round(num_class_model_3/int(segment_goal)*100)}%')
    bar_2 = st.progress(int(num_class_model_3(segment_goal)*100))

with tabs[2]: #'Add Dataset'
    """In this tab, it is possible to see the ready and ongoing tasks from CVAT annotation tool.
    Also, you can download the annotations from completed tasks"""
    layout_dataset = st.columns([2, 2], gap = 'small')
    #two columns: left one represents the projects, right one represents the tasks within those projects
        
    with layout_dataset[0]:
        #list all cvat projects
        st.header('CVAT Projects')

        with ApiClient(configuration) as api_client:
            name_of_projects = []
            name_of_projects.append('None')
            list_of_projects = api_client.projects_api.list()
            for i in list_of_projects[0]['results']:
                name_of_projects.append(i['name'])

            cvat_projects = st.selectbox('Please choose cvat project', name_of_projects)
            if cvat_projects!='None':
                for i in list_of_projects[0]['results']:
                    if cvat_projects == i['name']:
                        project_id = i['id']
                        project_task_num = len(i['tasks'])
                        st.write('Project Name: ', cvat_projects)
                        st.write('Created Date: ', str(i['created_date'])[:10])
                        st.write('Labels: ', str([z['name'] for z in i['labels']]))
                        st.write('Created by: ', i['owner']['username'])
                        st.write('Number of created tasks in this project: ', len(i['tasks']))
                        st.write('URL of project: ', f'{cvat_url}/projects/'+str(i['id']))

    with layout_dataset[1]:
        if cvat_projects!='None':
            st.header('"'+cvat_projects + '" Tasks: ')
            list_of_project_tasks = api_client.projects_api.list_tasks(id=project_id, page_size = project_task_num)
            
            name_of_tasks = [i['name'] for i in list_of_project_tasks[0]['results']]

            cvat_project_tasks = st.selectbox('Please choose task ', name_of_tasks)
            for i in list_of_project_tasks[0]['results']:
                    if cvat_project_tasks == i['name']:
                        project_task_id = i['id']
                        try:
                            assignee = i['assignee']['username']
                        except:
                            assignee = i['assignee']
                        # project_task_n = len(i['tasks'])
                        st.write('Task Name: ', cvat_project_tasks)
                        st.write('Created Date: ', str(i['created_date'])[:10])
                        st.write('Number of images: ', i['segment_size'])
                        st.write('Created by: ', i['owner']['username'])
                        st.write('Assigned to: ', assignee)
                        st.write('Status: ', i['status'].upper())
                        st.write('Last edited time: ', str(i['updated_date'])[:10])
                        st.write('URL of project: ', 'your_cvat_url/tasks/'+str(i['id']))

            (list_of_jobs_in_task,response )= api_client.tasks_api.list_jobs(id=project_task_id)
            
            job_id = list_of_jobs_in_task[0]['id']

            #You can download any task's annotations in any format, here we download in COCO format using cvat job id
            st.header(f'Download Annotations of task {cvat_project_tasks} in COCO Format')
            if st.button('Download Annotations'):

                status_code = -1
                i=1
                while status_code != 200:
                    output = session.get(f'{cvat_url}api/jobs/{job_id}/annotations/?action=download&filename=hello&format=COCO%201.0&location=local&use_default_location=true')
                    status_code = output.status_code
                    st.write(status_code)
                    if i>=100:
                        st.write('Sorry, could not download annotations, try again later')
                        break
                    i +=1
                #download into zip file
                with open(f"{cvat_project_tasks}_annotations.zip", "wb") as f:
                    f.write(output.content)
                if os.path.isfile(f"{cvat_project_tasks}_annotations.zip"):
                    with zipfile.ZipFile(f"{cvat_project_tasks}_annotations.zip","r") as zip_ref:
                        zip_ref.extract('annotations/instances_default.json')

                #move annotations into required folder with specified name
                shutil.move('annotations/instances_default.json', f"downloaded_annotations/project/{cvat_project_tasks}_annotations.json")
                os.remove(f"{cvat_project_tasks}_annotations.zip")
                shutil.rmtree('annotations')
                st.write(f'Now you may add file "{cvat_project_tasks}_annotations.json" to existing dataset (section below)')
        
            st.header('Add new annotations to existing dataset')

            can_train = False
            
            #If you have your own annotation files, it is possible to add them also
            select_how_to_upload = st.selectbox('Please choose how to upload files: ', ['From server', 'From personal computer'])
            if select_how_to_upload == 'From personal computer':
                with st.form('uploading form', clear_on_submit= True):
                    uploaded_files = st.file_uploader("Please choose a file or multiple files", accept_multiple_files=True)
                    saved_uploaded_files = uploaded_files.copy()
                    submitted = st.form_submit_button('Add to current dataset')
            
                    if uploaded_files:
                        st.write('You uploaded this file(s):')
                        for i in uploaded_files:
                            st.write(i.name)
                        for file in uploaded_files:
                            with open(f'downloaded_annotations/{file.name}', mode='wb') as w:
                                w.write(file.getvalue())

            if select_how_to_upload == 'From server':
                choose_project = st.selectbox('Choose sub project', ['SubProject_1', 'SubProject_2'])
                if choose_project == 'SubProject_1':
                    uploaded_files = st.multiselect('Choose files to add', os.listdir('downloaded_annotations/subproject_1'))
                else: uploaded_files = st.multiselect('Choose files to add', os.listdir('downloaded_annotations/subproject_2'))
                
            if uploaded_files:
                if st.button('add to dataset'):
                    st.write('removing current dataset folder')
                    st.write('.............................')
                    if choose_project == 'SubProject_1':
                        num_of_files = len(os.listdir('automated_datasets'))
                        total_num_files_in_model = len(os.listdir('dataset_subproject_1/dataset/train'))+len(os.listdir('dataset_subproject_1/dataset/test'))
                        shutil.move('dataset_subproject_1',f'/automated_datasets/dataset_subproject_1_{str(date.today())}_{str(total_num_files_in_model)}_{str(num_of_files)}')
                        os.mkdir('dataset_subproject_1')
                        # os.mkdir('dataset_stenosis/dataset')
                        os.mkdir('dataset_subproject_1/weights')
                        
                    elif choose_project == 'SubProject_2':
                        num_of_files = len(os.listdir('automated_datasets'))
                        total_num_files_in_model = \
                        len(os.listdir('dataset_subproject_2/dataset/train'))+len(os.listdir('dataset_segmentation/dataset/test')) 
                    
                        shutil.move('dataset_subproject_2',f'automated_datasets/dataset_subproject_2_{str(date.today())}_{str(total_num_files_in_model)}_{str(num_of_files)}')
                        os.mkdir('dataset_subproject_2')
                        os.mkdir('dataset_subproject_2/dataset')
                        os.mkdir('dataset_subproject_2/weights')

                    st.write('.............................')
                    st.write('current dataset folder is moved to git project [see folder github_repo]. Please move folder to dvc')
                    #it is possible to run the bash scripts to send folders to dvc/git
                    list_of_files_uploaded = [i[:-5] for i in uploaded_files]
                    st.write(list_of_files_uploaded)
                    can_train = True
                    
                    
                        
                    all_join_divide(list_of_files_uploaded, 'new_dataset', choose_project+str(date.today()))
                    st.write('Hooray, your files are added to dataset, you can now retrain your dataset')
                    
with tabs[3]: #Train
        st.header('Current training parameters: ')
        

        url_mlflow = 'mlflow_url/#/experiments/0'

        st.markdown(f'''
                    <a href={url_mlflow}><button style="background-color:LightBlue; font-size: 36px">MLFLOW</button></a>
                    ''', unsafe_allow_html=True)

        st.header('Train')

        project_to_train = st.selectbox('Please select subproject to train', ['SubProject_1','SubProject_2'])
        
        st.write(f'You selected {project_to_train} project')
        st.write('Please enter parameters:')
        num_iter = st.text_input("Number of iterations: ", 10000)
        weight_folder_name = st.text_input("Name of model weights folder (Separate with only underscore): ", 'weights_folder')
        experiment_name = st.text_input("Name of experiment (Separate with only underscore): ", 'experiment_1')
        if project_to_train == 'SubProject_1':
            whole_train_folder = 'dataset_subproject_1/dataset'
            num_class = 2
        elif project_to_train == 'SubProject_2': 
            whole_train_folder = 'dataset_subproject_2/dataset'
            num_class = 27


        if st.button('Start Training'):
            to_continue = True
            if project_to_train == 'SubProject_1' and f'dataset_subproject_1/{weight_folder_name}' in os.listdir('dataset_subproject_1'):
                st.write('This weights folder already exists, please choose another name')
                to_continue = False
            if project_to_train == 'SubProject_2' and f'dataset_subproject_2/{weight_folder_name}' in os.listdir('dataset_subproject_2'):
                st.write('This weights folder already exists, please choose another name')
                to_continue = False
            if weight_folder_name == 'weights_folder' or experiment_name == 'experiment_1' or to_continue==False:
                st.write('Please choose other names for weights folder and experiment name')
            else: 
                st.write('Training has started')
                #Here it is possible to start training with chosen parameters
                #make sure to deploy training urls
                try:
                    x = requests.get(f'subproject_1_training_url/start_training/?whole_train_folder={whole_train_folder}&iter={num_iter}&num_class={num_class}&output_folder={weight_folder_name}&exp_name={experiment_name}&batch_size=64')
                except:
                    x = requests.get(f'subproject_1_training_second_url/start_training/?whole_train_folder={whole_train_folder}&iter={num_iter}&num_class={num_class}&output_folder={weight_folder_name}&exp_name={experiment_name}&batch_size=64')
                

                st.write(x.status_code)

# For ease of tracking task updates or hypothesis or simple comments -> I integrated simple update tracking module
            
with tabs[4]: #Tasks&Hypothesis&Updates

        st.header('Here are the hypothesis that are checked')
        st.write('')
        url_linear = "https://api.linear.app/graphql"

        headers = {
            'Content-Type': 'application/json',
            f'Authorization': 'Bearer {linear_token}',
        }

        data = '{"query" : "{ issues { nodes { title project {name} assignee { name } dueDate  state { name } } } }"}'
        

        
        response = requests.post(url=url_linear, headers=headers, data=data)
        response.encoding = 'text'
        if response.status_code == 200:
            my_dict = response.content.decode('utf-8')
            res = json.loads(my_dict)

        else: st.write( response.status_code, response.text)

        nodes = res['data']['issues']['nodes']
        layout_linear = st.columns([2, 2], gap = 'small')
        df_tasks_1 = pd.DataFrame(columns = ['Task Name','Assignee','Status', 'Due Date'])
        df_tasks_2 = pd.DataFrame(columns = ['Task Name','Assignee','Status', 'Due Date'])
        with layout_linear[0]:
            #Let's consider only current week tasks that are needed to be accomplished
            #TO modify - add only working current week (if today is friday, show only friday tasks, currently it shows tasks till next friday)
            st.header('Current Week Tasks')

            for i in nodes:
                

                if  datetime.strptime(i['dueDate'], '%Y-%m-%d').date() < date.today()+ timedelta(7) and i['state']['name']!='Done':
                    try:  
                        #show required tasks
                        df_dictionary_1 = pd.DataFrame([{
                            'Task Name':i['title'], 
                            'Assignee':i['assignee']['name'], 
                            'Status':i['state']['name'], 
                            'Due Date':i['dueDate']
                            }])
                        df_tasks_1 = pd.concat([df_tasks_1, df_dictionary_1], ignore_index = True)
                    
                    except: continue
                     
                else: continue

            st.write(df_tasks_1)

        with layout_linear[1]:
            # Some tasks may lack due date or assignee, I decided to list them also
            st.header('Without Due Date or Without Assignee Tasks')

            for i in nodes:
                if i['dueDate']==None or i['dueDate']=="None" and i['state']['name']!='Done' :
                    try: 
                        if i['state']['name']!='Done' and len(i['state']['name'])<11 :
            
                            df_dictionary_2 = pd.DataFrame([{
                              'Task Name':i['title'],
                              'Assignee':i['assignee']['name'], 
                              'Status':i['state']['name'], 
                              'Due Date':i['dueDate']
                              }])
                            df_tasks_2 = pd.concat([df_tasks_2, df_dictionary_2], ignore_index = True)   
                    except: #if no assignee name:
                        if i['state']['name']!='Done' and len(i['state']['name'])>11 :
                            
                            df_dictionary_2 = pd.DataFrame([
                                {'Task Name':i['title'],
                             'Assignee':i['assignee'],
                              'Status':i['state']['name'],
                               'Due Date':i['dueDate']}
                               ])
                            df_tasks_2 = pd.concat([df_tasks_2, df_dictionary_2], ignore_index = True)

            st.write(df_tasks_2)    

        # Now we can track project updates
        st.header('Project Updates')
        #Make sure to create updates json with key as str date and value is str text
        f = open('updates.json')
        all_updates = json.load(f)

        list_of_updates = list(all_updates.keys())
        #list all updates in reverse order - making latest to appear first
        for i in range(len(list_of_updates)-1, -1, -1):
            st.write(list_of_updates[i], ': ', all_updates[list_of_updates[i]])

        update_input = st.text_input('Please Enter your Update below', key = 'update_creation')
        if update_input:
            st.write('You entered update: ', update_input)
            if update_input.lower() in [x.lower() for x in list(all_updates.values())]:
                    st.write('Update already exists, Please enter new update')
            else:
                if st.button('Update Project Updates'):
                    
                        if str(date.today()) in list(all_updates.keys()):
                            all_updates[str(date.today())+'_'+str(len(all_updates))] = update_input
                        else: all_updates[str(date.today())] = update_input
                        with open('updates.json','w') as f:
                            json.dump(all_updates, f)

        # it is done to update the page.
        st.button('update updates bar', on_click =clear_updates)

with tabs[5]: #Inference on single image
    #You can modify this tab to inference on whole folder
    #Or you insert the groundtruth - prediction checking 
    the_very_start = time.time()
    st.header('Please evaluate your model on single image')
    subproject_1_weights_folder = st.selectbox('Please choose weights folder for SubProject_1 ', os.listdir('dataset_subproject_1/weights'))
    subproject_2_weights_folder = [z for z in os.listdir('dataset_subproject_2/weights') if z!='dataset']

    model_weights_folder_subproject_2 = st.selectbox('Please choose weights folder for SubProject_2 ', subproject_2_weights_folder)
    threshold = st.text_input("choose threshold: ", 0.5)

    st.title('Detection')
    st.header('Upload the image to analyze')

    uploaded_file = st.file_uploader("Choose a file to compare two models")
    if st.button('Predict Image'):

        with open('temp_folder/new_uploaded_image.png', mode='wb') as w:
            w.write(uploaded_file.getvalue())

        img = cv2.imread('temp_folder/new_uploaded_image.png')

        #INSERT HERE YOUR INFERENCE SCRIPT FOR EACH OF SubProjects
        #st.image(plot(predict(img)))
    

with tabs[6]: #Check between models
    #Modify previous tab to compare between two models or weights
    pass

    
