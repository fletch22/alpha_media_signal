REM git clone https://github.com/fletch22/alpha_media_signal.git C:\Users\Chris\deploy_environments\prod\ams

c:

cd \Users\Chris\deploy_environments\prod\ams

git checkout master

REM reset this repository's master branch to the commit of interest
git reset 44d4de6 --hard

cd \Users\Chris\workspaces\alpha_media_signal\scripts\deploy\