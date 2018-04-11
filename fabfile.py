from fabric.api import *
from fabric.contrib import *
from fabric.contrib.project import rsync_project

def prod():
    env.hosts = ['deploy@10.9.61.2']
    env.app_directory = '/mnt/Detectron'
    env.role = 'prod'
    env.stopcmd = 'supervisorctl stop api'
    env.restartcmd = 'supervisor restart api'

def rsync_project_internal(local_dir, remote_dir, **kwargs):
    project.rsync_project(local_dir=local_dir, remote_dir=remote_dir, ssh_opts="-o StrictHostKeyChecking=no", **kwargs)

def deploy():
    rsync_project_internal(local_dir="api conf tools lib demo",
                           remote_dir=env.app_directory, exclude=["*~", "*.pyc"])


def restart():
    with cd(env.app_directory):
        sudo(env.restartcmd)

def all():
    prod()
    deploy()
    restart()

if __name__ == '__main__':
    all()