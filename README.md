# COVID-19-Misinformation
My project repo for Data Sciences Fellowship Program on 2022 Summer

COVID-19 was claimed as a worldwide pandemic on March 11, 2020. Misleading	COVID-19  information can cause people’s fear such as panic-buying, vaccine hesitancy, crash market, as well as mask-wearing hesitancy. Timely detection the misleading information on social media can prevent its spread to a large population which might cause huge impacted problems.

# Dataset
I reused the manually labeled datasets from these two research groups, which contain tweet IDs and whether they were identified as misinformation or not. 

1. [CoAID](https://github.com/cuilimeng/CoAID): COVID-19 heAlthcare mIsinformation Dataset

- background: COVID-19 virus quickly spreads around the world, and so does for the COVID-19 misinformation. Such information has caused confusion among people, disruptions in society, and even deadly consequences in health problems. The auhtors collected and annotated the COVID-19 related news and claims on websites, social platforms.
- how they labeled data: Claim was annotated into fake or true by comparing the reliable sources, WHO, MNT. They classified the myth and rumor as fake claim'. Some examples: "Only older adults and young people are at risks" is a fake claim, while "5G mobile networks DO NOT spread COVID-19" is a true claim 
- chosen dataset: 457 fake claim(of one or two sentences) tweets, 6342 real calim tweets which collected September 1, 2020 from through Nov 1, 2020

2. [AnTiVax](https://github.com/SakibShahriar95/ANTiVax)

- background: There has been a rise in vaccine hesitancy beacuse of misinformation being spread on social media. To promote research in COVID-19 vaccine misinformaiton detection work, the authors of this paper shared their collected and annotated COVID-19 vaccine tweets to public.
- how they labeled data: classfied into misinformation or general vaccine tweets using reliable source by human and validated by medical experts. This ensured the sarcastic and humorous content were not included as misinformation.
- chosen dataset: They used these keywords to extract COVID-19 vaccine related dataset from Dec 1, 2020 to  July 31, 2021: 'vaccine', 'pfizer', 'moderna', 'astrazeneca', 'sputnik' and 'sinopharm'. Finally, 15, 073 tweets were labeled, 5751 of them were misinformation and 9322 were general vaccine-related tweets.


# Instruction
We have following folders:

- [`Data`]()

| Variable | Description |
| --- | --- |
| index | de-identified person ID (only CoAID dataset has) |
| id | tweet ID |
| is_missinfo| 1 if this is misinformation; 0 if not |

- [`Preprocess`]()
- [`Model`]()
- [`Demostration`]()

# Used Open-sourced Tool/API

## Twitter API

Twitter data is gathered using Twitter Developer account and API keys. The twitter developer account can be created at [website](https://developer.twitter.com/en). Once the account is created, you can create the app. On successful creation of the app, the keys will be  available in the `keys and tokens` section of the app.
  
## Hydrator

Developers can't share the detialed individual-level twitter data based on privacy policy. However, the tweets IDs are allowed to share. [Hydrator](https://github.com/DocNow/hydrator) is a desktop application for hydrating Twitter ID datasets which can turn tweets IDs back into JSON or CSV complete files.
