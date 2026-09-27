Tutorial for Using Compute Canada Resources

1. you need to create an account under someone's domain. I was with ombuki so i would be under hers. you submit your account creation, it gets approved. now you login using git bash or something.

2. in git bash, type ssh `username@resource.computecanada.ca`. replace `username` with the username you used for your account reaction and resource for the supercomputer you want to use e.g `fir`. you are then prompted to enter your password and you also have to download an app called `duo mobile` for 2fa. its basically microsoft authenticator type, when you sign in it'll ask you if you're signing in you just hit yes and you're logged in.
3. from there you just clone a github repo of whatever you want it to run.

4. you have to set up this thing called a slurm script, its how the computer takes in python files. but before that you should set up a virtual environment assuming you're gonna
use libraries like numpy. see `setup_VE.txt` for instructions on how to do that. once you're done that you can now upload python scripts that need libraries and they should run
just fine, but you can skip this step if you're just using native Python.

5. now you can create a slurm script. these are needed for running python scripts; you cant just send your script to the computer to crunch right away as other people are using it and there could be no processors available so this slurm stuff handles all that. see `job.txt` to get code for the slurm script. get that code and `nano` from your main project directory and create a file called `job.sh`. paste the code from `job.txt` in there and save. know that the top shebang lines in the slurm script are essential. especially the `#SBATCH account` one. you need to put your superiors' account there for any slurm scripts to be accepted. the other ones serve as parameters to the script, essentially. the notable ones are `time` which sets the duration for your execution. you usually want to set enough time for your program to run, because it will stop mid-execution if you didnt allocate enough time. you can also specify memory and how many processors per task you want, among others. go research the other parameters for the slurm script, there's plenty more.

6. after your slurm script is created, type `sbatch job.sh` and it'll queue your job to the supercomputer. you have to wait some time for your job to start executing. you can type `squeue` to see the entire queue of items but you can type `sq` to see your stuff in the queue.

7. when your slurm script is finished, an output file will be created in the same directory, default named after the job id. you can find anything printed to console there. you can also rename the output files (there's a parameter for that).