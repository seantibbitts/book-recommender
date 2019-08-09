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

You will need to install the original lightfm package. Follow the directions [here](https://github.com/lyst/lightfm).

Then, once you have cloned the repo, cd into functions/lightfm_ext and run `pip install .`.
This will install the 'lightfm_ext' package, which underlies the 'Rate-Your-Own Recommender.'

## Running

To run, cd into the top level of the repository and run `python script-all.py `. This will launch the Flask app,
and will allow you to play around with the recommenders.

## Next Steps

My original intention was to deploy this to Heroku, but it required too much RAM for the free tier. I then attempted
to deploy it to AWS Lambda with Zappa, but the app is apparently too large. I may end up deploying this to a static
AWS instance.
