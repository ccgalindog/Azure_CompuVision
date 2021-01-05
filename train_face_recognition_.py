import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import click

"""
This code creates a PersonGroup object in Azure Face API. This PersonGroup is a group of people that you want to be able to recognize
through their faces. After training over multiple images of each individual, an ID is assigned to each person so that it can
be referenced in the future.

Parameters:

KEY: The access key provided by Face API on Azure.
ENDPOINT: The link to the server where Face API runs, for instance: "https://myfacesbook.cognitiveservices.azure.com/"
PERSON_GROUP_ID: The title name for the PersonGroup.
IMG_FOLDER: The path where person folders can be found; each of this folders has to contain multiple images of a particular person and is named after said person.

After this model has been trained, you should be able to perform inferences on new images and recognize whether a person in this PersonGroup is there or not.
"""


@click.command()
@click.option('--KEY', default='', help='Key to Face API.')
@click.option('--ENDPOINT', default='', help='Endpoint to Face API')
@click.option('--PERSON_GROUP_ID', default='', help='Desired name for the person group.')
@click.option('--IMG_FOLDER', default='Images/Train_Images', help='Path to the folder that contains subfolders with images.')

def main( KEY, ENDPOINT, PERSON_GROUP_ID, IMG_FOLDER):

	face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY)) 

	TARGET_PERSON_GROUP_ID = str(uuid.uuid4())

	face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)


	all_folders = glob( IMG_FOLDER + '/*' )
	all_person_names = [ some_folder.split('/')[-1] for some_folder in all_folders ]

	for each_person in all_person_names:
		dis_person_id = face_client.person_group_person.create(PERSON_GROUP_ID, each_person)
		dis_person_imgs = [file for file in glob.glob('{}/{}/*'.format( IMG_FOLDER, each_person ) ) ]

		for some_image in dis_person_imgs:
			w = open( some_image, 'r+b')
		try:
			face_client.person_group_person.add_face_from_stream( PERSON_GROUP_ID, dis_person_id.person_id, w )
		except:
			pass


	print('Training the person group...')
	face_client.person_group.train(PERSON_GROUP_ID)

	while (True):
		training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
		print("Training status: {}.".format(training_status.status))

		if (training_status.status is TrainingStatusType.succeeded):
			break
		elif (training_status.status is TrainingStatusType.failed):
			sys.exit('Training the person group has failed.')
		time.sleep(5)

	print('Person group: {} has been trained. You can now use it for inference.'.format( PERSON_GROUP_ID ) )


if __name__ == '__main__':
	main()