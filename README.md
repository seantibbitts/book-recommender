# Book Recommender

This repository holds the code of three different book recommendation engines.

- Goodreads Recommenders:

  These pull reviews from any public Goodreads user account and generate recommendations based on those.

  - Implicit Recommender
  - Explicit Recommender
- Don't have a Goodreads account? Rate books and get recommendations.
  - Rate-Your-Own Recommender

These recommendation engines were built on the [Goodreads](https://www.goodreads.com/) API,
the [LightFM recommender model](https://github.com/lyst/lightfm) by Maciej Kula
and the [goodbooks-10k book dataset](https://github.com/zygmuntz/goodbooks-10k) by Zygmunt Zając.

## Installation

Once you have cloned the repo
- Create a python virtual environment in whatever way you prefer
- Activate the virtual environment
- Run `pip install Cython`
- cd into the top directory of the repo and run `pip install -r requirements.txt`
This should install all of the dependencies in your virtual environment, including the 'lightfm_ext' package, which underlies the 'Rate-Your-Own Recommender.'

To run the Goodreads recommenders, you will need to sign up for a [Goodreads API key](https://www.goodreads.com/api/keys).
You will then need to either pass the api key to the script at runtime or set the api key as an environmental variable
called 'GOODREADS_API_KEY' in your virtual environment. Alternatively, you can edit script-all.py to pull from a different
source (such as keyring).

## Running

To run, cd into the top level of the repository and run `python script-all.py <api_key>`. (If you set the api key
as an environmental variable, you can just run `python script-all.py`.) This will launch the Flask app,
and will allow you to play around with the recommenders.

## Next Steps

My original intention was to deploy this to Heroku, but it required too much RAM for the free tier. I then attempted
to deploy it to AWS Lambda with Zappa, but the app is apparently too large. I may end up deploying this to a static
AWS instance.

This use case (an online recommender that can handle new users but not new books) might be better suited to an item-item
similarity recommender system. I will look into that.
