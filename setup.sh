#Export Python Path
export PYTHONPATH=/home/duypd/ThisPC-DuyPC/SG-Retrieval:$PYTHONPATH
echo $PYTHONPATH

#Generate Entities 
# python3 Entities/entities.py

#Run API
# python3 app.py

# CUDA_VISIBLE_DEVICES=0, python3 Controller/IRESGController/train.py 

export JWT_SECRET_KEY="This_is_my_sceret_key" #linux

setx JWT_SECRET_KEY "This_is_my_sceret_key" #win
$env:JWT_SECRET_KEY="This_is_my_sceret_key" #win