# Sith_Predictor
An algorithm which predicts if a being will become a sith. 82% acccuracy, enough to save some younglings.

After scraping Star Wars Characters (https://starwars.fandom.com/wiki/Databank_(original)) with  BeautifullSoup(bs4) for all the users and importing them into a Dataframe, I used XGBoosting (Ensemble Method using Supervised Machine Learning) to work on 646 different characters to predict who would or wouldnt affiliate with the Galactic Empire, 70% of them were used for training and 30% for testing.

Thedatabase has the following columns:
1. 'Affiliation(s)' of the character.
2. 'Apprentices'
3. 'Average height'
4. 'Average length'
5. 'Average lifespan'
6. 'Average mass'
7. 'Caste'
8. 'Clan(s)' for which the character has been a member of.
9. 'Cybernetics' the character carries.
10. 'Designation'
11. 'Distinctions'
12. 'Domain'
13. 'Duties'
14. 'Established by'
15. 'Eye color'
16. 'Government'
17. 'Hair color'
18. 'Height'
19. 'Homeworld'
20. 'Kajidic' is a Hutt social structure that was both a family and a crime gang
21. 'Language'
22. 'Mass'
23. 'Masters'
24. 'Organization'
25. 'Skin color'
26. 'Species'
