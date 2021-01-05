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
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person



"""
This code uses an existing PersonGroup to perform inferences on new images.
For each testing image passed, a new output image is generated with the corresponding tags for each person,
including his/her approximate age, mood and name, if recognized. 

Parameters:

KEY: The access key provided by Face API on Azure.
ENDPOINT: The link to the server where Face API runs, for instance: "https://myfacesbook.cognitiveservices.azure.com/"
PERSON_GROUP_ID: The title name of the PersonGroup you already trained.
TEST_IMG_FOLDER: The path to the folder which contains all testing images.
OUTPUT_FOLDER: Path where output (tagged) images will be saved.

After this model has been trained, you should be able to perform inferences on new images and recognize whether a person in this PersonGroup is there or not.
"""

def getRectangle(faceDictionary ):
	rect = faceDictionary.face_rectangle
	left = rect.left
	top = rect.top
	right = left + rect.width
	bottom = top + rect.height

	return ((left, top), (right, bottom))

def get_emotion(emoObject):
	emoDict = dict()
	emoDict['anger'] = emoObject.anger
	emoDict['contempt'] = emoObject.contempt
	emoDict['disgust'] = emoObject.disgust
	emoDict['fear'] = emoObject.fear
	emoDict['happiness'] = emoObject.happiness
	emoDict['neutral'] = emoObject.neutral
	emoDict['sadness'] = emoObject.sadness
	emoDict['surprise'] = emoObject.surprise
	emo_name = max(emoDict, key=emoDict.get)
	emo_level = emoDict[emo_name]
	return emo_name, emo_level




@click.command()
@click.option('--KEY', default='', help='Key to Face API.')
@click.option('--ENDPOINT', default='', help='Endpoint to Face API')
@click.option('--PERSON_GROUP_ID', default='', help='Desired name for the person group.')
@click.option('--TEST_IMG_FOLDER', default='Images/Test_Images', help='Path to the folder that contains testing images.')
@click.option('--OUTPUT_FOLDER', default='Outputs', help='Path to the folder where output images will be saved.')

def main( KEY, ENDPOINT, PERSON_GROUP_ID, TEST_IMG_FOLDER, OUTPUT_FOLDER):

	face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

	test_image_array = glob.glob('{}/*'.format(TEST_IMG_FOLDER) )

	for each_testimg in test_image_array:

		image = open(each_testimg, 'r+b')

		time.sleep( 60 )

		face_ids = []

		faces = face_client.face.detect_with_stream(image, detectionModel='detection_02',
									return_face_attributes = ['emotion', 'age', 'gender'])


		for face in faces:
			if len(face_ids) < 9:
				face_ids.append(face.face_id)


		results = face_client.face.identify(face_ids, PERSON_GROUP_ID)

		found_people = {}

		print('Identifying faces in {}'.format(os.path.basename(image.name)))
		if not results:
			print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
		for person in results:
			if len(person.candidates) > 0:

				name_person = face_client.person_group_person.get( PERSON_GROUP_ID, person.candidates[0].person_id )
				found_people[ person.face_id ] = [name_person.name, person.candidates[0].confidence]
				
				print('Person for face ID {} is identified in {} to be {} with a confidence of {}.'.format(person.face_id,
								os.path.basename(image.name), name_person.name, person.candidates[0].confidence)) 
			else:
				found_people[ person.face_id ] = ['Unknown', -1]
				print('No person identified for face ID {} in {}.'.format(person.face_id, os.path.basename(image.name)))



		img = Image.open( test_image_array[0] )
		draw = ImageDraw.Draw(img)
		fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 80)
		for face in faces:
			try:
				dis_list_results= found_people[ face.face_id ]
				bounding_rect = getRectangle(face)
				draw.rectangle( bounding_rect, outline='crimson', width = 2)
				emotion, confidence = get_emotion(face.face_attributes.emotion)
				di_age = face.face_attributes.age
				di_gender = face.face_attributes.gender
				if dis_list_results[0] != 'Unknown':
					draw.text( bounding_rect[0] , dis_list_results[0] + '  {:.2f}%'.format( 100*dis_list_results[1] ), fill=(255,255,255,255))
				else:
					draw.text( bounding_rect[0] , dis_list_results[0], fill=(255,255,255,255))
				draw.text( bounding_rect[1] , emotion + '  {:.2f}%'.format( 100*confidence ) + ' Age: {} '.format(di_age) + di_gender, fill=(255,255,255,255))	
			except:
				pass
		#img.show()
		img.save("{}/Tagged_{}_.png".format( OUTPUT_FOLDER, test_image_array[0].split('/')[-1].split('.')[0] ) )