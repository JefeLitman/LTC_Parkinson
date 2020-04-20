import tensorflow as tf

def ct_covid():
    """Function to get the encoding dictionary for CT_COVID.tfrecord"""
    encode_dict = {
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'is_covid': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'img': tf.io.FixedLenFeature([], tf.string),
    }
    return encode_dict

def xr_chexpert():
    """Function to get the encoding dictionary for XR_CheXpert.tfrecord"""
    encode_dict = {
        'patient_id': tf.io.FixedLenFeature([], tf.string),
        'study_number': tf.io.FixedLenFeature([], tf.string),
        'sex': tf.io.FixedLenFeature([], tf.string),
        'age': tf.io.FixedLenFeature([], tf.int64),
        'view': tf.io.FixedLenFeature([], tf.string),
        'view_type': tf.io.FixedLenFeature([], tf.string),
        'diseases': tf.io.FixedLenFeature([8], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'img': tf.io.FixedLenFeature([], tf.string),
    }
    return encode_dict

def radiopaedia():
    """Function to get the encoding dictionary for RadioPaedia.tfrecord"""
    encode_dict = {
    'patient_id': tf.io.FixedLenFeature([], tf.int64), # The patient id related to the excel metadata (only integers values)
    'sex': tf.io.FixedLenFeature([], tf.string), # The genre of the person 'male', 'female' or 'unknow' values
    'age': tf.io.FixedLenFeature([], tf.int64), # The age of the patient, if the value is -1 is unknow
    'view_type': tf.io.FixedLenFeature([], tf.string), # Can be 'axial' or 'coronal'
    'have_noise': tf.io.FixedLenFeature([], tf.string), # If the image have noise ('device' or 'unknow' values)
    'is_covid': tf.io.FixedLenFeature([], tf.int64), # Always 1 because this data have only covid cases
    'frames': tf.io.FixedLenFeature([], tf.int64), # Total numbers of images in the CT
    'height': tf.io.FixedLenFeature([], tf.int64), # Height of the image
    'width': tf.io.FixedLenFeature([], tf.int64), # Width of the image
    'channels': tf.io.FixedLenFeature([], tf.int64), # Channels of the image, generally is 3
    'ct_volumen': tf.io.FixedLenFeature([], tf.string), # The volumen data itself
    }
    return encode_dict