# Helper class for DALL-E
# The following class creates a simple wrapper on the Azure OpenAI REST endpoints. It will simplify the steps for calling the text-to-image API to submit your request and then poll for the results

import requests
import time

class ImageClient:
    def __init__(self, endpoint, key, api_version = "2022-08-03-preview", verbose=False):
        # These are the paramters for the class:
        # ### endpoint: The endpoint for your Azure OpenAI resource
        # ### key: The API key for your Azure OpenAI resource
        # ### api_version: The API version to use. This is optional and defaults to the latest version
        self.endpoint = endpoint
        self.api_key = key
        self.api_version = api_version
        self.verbose = verbose

    def text_to_image(self, prompt):
        # this method makes the text-to-image API call. It will return the raw response from the API call

        reqURL = requests.models.PreparedRequest()
        params = {'api-version':self.api_version}
        #the full endpoint will look something like this https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/dalle/text-to-image
        reqURL.prepare_url(self.endpoint + "dalle/text-to-image", params) 
        if self.verbose:
            print("Sending a POST call to the following URL: {URL}".format(URL=reqURL.url))

        #Construct the data payload for the call. This includes the prompt text as well as many optional parameters.
        payload = { "caption": prompt}

        r = requests.post(reqURL.url, 
            headers={
                "Api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json = payload
        )
        # Response Body example: { "id": "80b095cb-4248-4fa7-90c2-933f0907fb2a", "status": "Running" }
        # Key headers:
        # Operation-Location: URL to get response
        # Retry-after: 3 //seconds to respond

        if r.status_code != 202:
            print("Error: {error}".format(error=r.json()))

        data = r.json()
        if self.verbose:
            print('text-to-image API response body:')
            print(data)
        return r

    def getImageResults(self, operation_location):    
        # This method will make an API call to get the status/results of the text-to-image API call using the
        # Operation-Location header from the original API call
        
        params = {'api-version':self.api_version}
        # the full endpoint will look something like this 
        # https://YOUR_RESOURCE_NAME.openai.azure.com/dalle/text-to-image/operations/OPERATION_ID_FROM_PRIOR_RESPONSE?api-version=2022-08-03-preview
      
        if self.verbose:
            print("Sending a POST call to the following URL: {URL}".format(URL=operation_location))

        r = requests.get(operation_location,
            headers={
                "Api-key": self.api_key,
                "Content-Type": "application/json"
            }
        )

        data = r.json()

        if self.verbose:
            print('Get Image results call response body')
            print(data)
        return r

        # Sending a POST call to the following URL: <operatino-location>
        # {'id': 'd63fc675-f751-40b7-a297-e692c3b966b9', 'result': {'caption': 'An avocado chair.', 'contentUrl': '<image location>', 'contentUrlExpiresAt': '2022-08-13T22:52:45Z', 'createdDateTime': '2022-08-13T21:50:55Z'}, 'status': 'Succeeded'} 


    def getImage(self, contentUrl):
        # Download the images from the given URL
        r = requests.get(contentUrl)
        return r


    def generateImage(self, prompt):
        submission = self.text_to_image( prompt)
        if self.verbose:
            print('Response code from submission')
            print(submission.status_code)
            print('Response body:')
            print(submission.json())
        if submission.status_code == 202:
            operation_location = submission.headers['Operation-Location']
            retry_after = submission.headers['Retry-after']
        else:
            print('Not a 202 response')
            return "-1"

        #wait to request
        status = "not running"
        while status != "Succeeded":
            if self.verbose:
                print('retry after: ' + retry_after)
            time.sleep(int(retry_after))
            r = self.getImageResults(operation_location)
            # print(r.status_code)
            # print(r.headers)
            # print(r.json())
            status = r.json()['status']
            # print(status)
            if status == "Failed":
                return "-1"
        
        contentUrl = r.json()['result']['contentUrl']
        image = self.getImage(contentUrl)
        return contentUrl, image.content



