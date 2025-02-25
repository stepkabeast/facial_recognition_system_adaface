from collections import namedtuple

Extensions = namedtuple('Parts', 'image_like_format '
                                 'pdf_like_format ')

recognized_mimes = Extensions(pdf_like_format=['application/pdf'],
                              image_like_format=['image/jpeg', 'image/png',
                                                 'image/tiff', 'image/x-ms-bmp', 'image/bmp']
                              )
