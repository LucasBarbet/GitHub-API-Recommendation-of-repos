import setuptools

with open('README.md','r',encoding="utf-8") as f:
    long_description=f.read()




__version__= "0.0.0"

REPO_NAME=''
AUTHOR_USER_NAME="lbarbet ateissier kpatault"
SRC_REPO = "GitHubAPIRecommendationOfRepos"
AUTHOR_EMAIL="lucas.barbet.Etu@univ-lemans.fr antoine.teissier.Etu@univ-lemans.fr kylian.patault.Etu@univ-lemans.fr"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='a small  python package for  mlops ',
    long_description=long_description,
    long_destription_content='text/markdown',
    url="https://github.com/LucasBarbet/GitHubAPIRecommendationOfRepos.git",
    project_urls={

        "Bug Tracker":"https://github.com/LucasBarbet/GitHubAPIRecommendationOfRepos.git/-/issues",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src'),


)
