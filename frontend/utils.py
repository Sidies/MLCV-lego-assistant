#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import flask
import http


def rest_property(getter, setter, type, key=None):
    """
    Handle the boilerplate of getting/setting a REST JSON property.
    This function handles GET and PUT requests for different datatypes.
    """
    # If the request method is GET, retrieve the value of the property using the getter function
    if flask.request.method == 'GET':
        value = getter()
        
        # If a key is provided, retrieve the value of the property using the key
        if key:
            value = value[key]
            
        # Return the value as a JSON response
        response = flask.jsonify(value)
    
    # If the request method is PUT, set the value of the property using the setter function
    elif flask.request.method == 'PUT':
        # Retrieve the new value of the property from the request JSON data and convert it to the specified type
        value = type(flask.request.get_json())
        
        # If a key is provided, set the value of the property using the key
        if key:
            setter(**{key:value})
        else:
            # Otherwise, set the value of the property directly
            setter(value)
            
        # Return an empty response with a 200 OK status code
        response = ('', http.HTTPStatus.OK)
        
    # Print a log message with the IP address, request method, request path, and value of the property
    print(f"{flask.request.remote_addr} - - REST {flask.request.method} {flask.request.path} => {value}")    

    # Return the response
    return response
        