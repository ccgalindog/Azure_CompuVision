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


# This key will serve all examples in this document.
KEY = 
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = 
# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


# Used in the Person Group Operations and Delete Person Group examples.
# You can call list_person_groups to print a list of preexisting PersonGroups.
# SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
PERSON_GROUP_ID = 'the_office_cast_3' # str(uuid.uuid4()) # assign a random ID (or name it anything)



'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against
test_image_array = glob.glob('Images/Scenes/scene_4_.jpg')
image = open(test_image_array[0], 'r+b')

print('Pausing for 60 seconds to avoid triggering rate limit on free account...')
time.sleep (60)

# Detect faces
face_ids = []
# We use detection model 2 because we are not retrieving attributes.
faces = face_client.face.detect_with_stream(image, detectionModel='detection_02')



for face in faces:
	if len(face_ids) < 9:
		face_ids.append(face.face_id)



# Identify faces



results = face_client.face.identify(face_ids, PERSON_GROUP_ID)



#print( face_ids )

found_people = {}

print('Identifying faces in {}'.format(os.path.basename(image.name)))
if not results:
	print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
for person in results:
	#print( person )
	if len(person.candidates) > 0:
		#found_u = person.face_id
		#certainty_u = person.candidates[0].confidence
		#print(person.candidates[0].person_id  )
		name_person = face_client.person_group_person.get( PERSON_GROUP_ID, person.candidates[0].person_id )
		found_people[ person.face_id ] = [name_person.name, person.candidates[0].confidence]
		
		print('Person for face ID {} is identified in {} to be {} with a confidence of {}.'.format(person.face_id,
						os.path.basename(image.name), name_person.name, person.candidates[0].confidence)) # Get topmost confidence score
	else:
		found_people[ person.face_id ] = ['Unknown', -1]
		print('No person identified for face ID {} in {}.'.format(person.face_id, os.path.basename(image.name)))


def getRectangle(faceDictionary ):
	rect = faceDictionary.face_rectangle
	left = rect.left
	top = rect.top
	right = left + rect.width
	bottom = top + rect.height

	return ((left, top), (right, bottom))






print('Drawing rectangle around face... see popup for results.')

img = Image.open( test_image_array[0] )
draw = ImageDraw.Draw(img)
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 80)
for face in faces:
	try:
		dis_list_results= found_people[ face.face_id ]
		bounding_rect = getRectangle(face)
		draw.rectangle( bounding_rect, outline='crimson', width = 2)
		if dis_list_results[0] != 'Unknown':
			draw.text( bounding_rect[0] , dis_list_results[0] + '  {:.2f}%'.format( 100*dis_list_results[1] ), fill=(255,255,255,255))
		else:
			draw.text( bounding_rect[0] , dis_list_results[0], fill=(255,255,255,255))
	except:
		pass
# Display the image in the users default image browser.
img.show()
img.save("Tagged_{}_.png".format( test_image_array[0].split('/')[-1].split('.')[0] ) )