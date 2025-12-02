# Papaya Spam Eater

- Python daemon that observes email maildir folders (filesystem watcher)
- Scans new mail upon arrival with feature extraction + neural net, selects category
- Either leaves mail in inbox or moves it to folder (Spam, Important, ...)
- Maintains a pointer to each inbox to remember last scanned mail (store to disk whenever it changes, reload on daemon start)
- For training we use mails from the category folders that are older than n days (let's start 7), assuming the users will have sorted them accordingly by then
