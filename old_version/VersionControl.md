# Git Terminal Commands

There are a few things going on in Git
- You have an online 'remote' repository which multiple people collaborate on, this contains code
- You have a 'main' branch which contains the best, bug-free version of the code
- You have branches, named 'develop/new-button-feature' etc which are environments to make changes
- You can work on a branch, then merge its changes to the main branch if no conflicts emerge
- On your local machine, you have an offline 'local' repository containing your version of the code
- You 'pull' code from remote -> local, and 'push' from local -> remote
- Each pull or push may have conflicts where the changes would cause big errors, and these need to be managed
- You also have the state of your current code on your local machine

- Do {Cmd + Shift + . } to show hidden files on a mac folder. Inside the folder with a local git repository you will see a hidden .git folder. This is where the Git application stores staged changes, commit logs etc, its not cloud based.


---------------------------------------
# Workflow

Setup
-----

- Setup a new repository on GitHub that is public or private, with a main branch maybe with a README file
- Create a work folder on your machine for the code, open it inside of VScode or other IDE
- Clone the existing remote repository into your work folder. This sets up a 'local' repository that is currently in sync with the remote one

Committing local changes
-----

- Within your working folder, create some new Python file and make some code, or change existing file's contents, and save all changes in each file
- Change to the right repository branch you want to commit to
- 'Stage' a set of changes made to either all files in the folder '.', or a specific file
- Commit these staged changes to the local repository, optionally adding a commit message
- Now 

Pushing changes
--------

- Create a new branch or move to an existing branch in local repository
- Commit local changes
- Pull from remote version of branch to see changes from other people before we push, ideally do this at start of development, respond to any conflicts
- Now push to the remote repository branch

Merging
------

- If branch is to be integrated into main, then move into that branch locally, making sure all changes pushed that we want, and that we're pulled and up to date
- Then merge branch into main, and check status

----------------------------------------

# Commands
(Run these in the terminal in the folder of interest)

- Initialise a local repository: \
```git init```

- Check the logs of the local repository \
``` git log```
``` git log --oneline``` (on one line)

- Clone remote GitHub repository into a new local folder, creating new repository: \
```git clone {GitHub URL} .``` \
(Note the . means it unpacks straight into the folder you are in (.), and doesn't create a subfolder)
If ```fatal : destination path '.' already exists ... ```, then don't do ```git init```, and clone the existing GitHub repo into EMPTY folder with no .git file.

- Stage changes for a file after its saved:\
```git add code.py```

- Stage all saved changes: \
``` git add .```

- Unstage changes: \
``` git reset ```

- Revert file to last committed stage: \
``` git checkout -- filename ```

- Commit all staged changes with message: \
```git commit -m "My Commit Message"```

- Check current status: \
```git status ```

- Create new local branch "develop/new-feature":\
``` git checkout -b develop/new-feature ```

- Change between branches: \
``` git checkout main ``` \
``` git checkout develop/new-feature ```

- Pull branch changes from remote repository: \
``` git pull origin develop/new-feature ```

- Push branch changes from local -> remote: \
``` git push origin develop/new-feature ```

- Merge branch into main: \
``` git checkout main ``` \
``` git merge develop/new-feature ```

- Show changes in unstaged files: \
``` git diff ```
``` git diff --staged``` (Show staged changes)
Note this takes you into an environment, press enter to keep scrolling
Press 'q' key to exit this environment back to terminal


