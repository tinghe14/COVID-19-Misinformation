# COVID-19-Misinformation
My project repo for Data Sciences Fellowship Program on 2022 Summer

COVID-19 was claimed as a worldwide pandemic on March 11, 2020. Misleading	COVID-19  information can cause people’s fear such as panic-buying, vaccine hesitancy, crash market, as well as mask-wearing hesitancy. Timely detection the misleading information on social media can prevent its spread to a large population which might cause huge impacted problems.

# Dataset
I reused the manually labeled datasets from these two research groups, which contain tweet IDs and whether they were identified as misinformation or not. 

1. [CoAID](https://github.com/gipplab/iConference22_COVID_misinformation)
2. [AnTiVax](https://github.com/SakibShahriar95/ANTiVax)

# Instruction
We have following folders:

- [`Data`]()
- [`Preprocess`]()
- [`Model`]()
- [`Demostration`]()

# Used Open-sourced Tool/API

## Twitter API

Twitter data is gathered using Twitter Developer account and API keys. The twitter developer account can be created at [website]
(https://developer.twitter.com/en). Once the account is created, you can create the app. On successful creation of the app, the keys will be  available in the `keys and tokens` section of the app.
  
## Hydrator

Developers can't share the complete twitter data based on privacy policy. However, the tweets IDs are allowed to share. [Hydrator](https://github.com/DocNow/hydrator) is a desktop application for hydrating Twitter ID datasets which can turn tweets IDs back into JSON or CSV complete files.


