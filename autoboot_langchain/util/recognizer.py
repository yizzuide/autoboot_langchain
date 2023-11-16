
import filetype

def detect_document_type(document_path):
    guess_file = filetype.guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    file_ext = guess_file.extension.lower()
    if(file_ext == "txt"):
      file_type = "txt"
    elif(file_ext == "md"):
      file_type = "md"
    elif(file_ext == "pdf"):
      file_type = "pdf"
    elif(file_ext == "doc" or file_ext == "docx"):
      file_type = "doc"
    elif(file_ext in image_types):
      file_type = "image"
    else:
      file_type = "unknown"
    return file_type