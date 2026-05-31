conda activate tfuse
export TRANSFUSER_ROOT=/ocean/projects/cis250201p/jjain2/transfuser
export PYTHONPATH=$PYTHONPATH:/jet/home/jjain2/Interpretable_Control
cd $CARLA_ROOT
./CarlaUE4.sh -opengl -RenderOffScreen -nosound &
python -c "import carla; client = carla.Client('localhost', 2000); client.set_timeout(10.0); print(client.get_server_version())"