from PIL import Image
import glob,os

files = glob.glob(os.path.join("C:\\Users\\adnan\\Desktop\\VAE\\NewTrack\\Feb22\\*.jpg"))
for f in files:
  try:
    image = Image.open(f)
  except OSError:
    print('Delete' + f)
    
  image = image.resize((160,120))
  image.crop((0, 40, 160, 120)).save(f, quality=95)
  
print("done")
print(files)
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
print("done")
