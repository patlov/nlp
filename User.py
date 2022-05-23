from dataclasses import dataclass


@dataclass
class User:
    user_id: int
    comments: list
    numberPositiveVotes: int
    numberNegativeVotes: int
    creationDate: str

    def addComment(self, comment: str):
        self.comments.append(comment)

    def setPositiveVotes(self, votes: int):
        self.numberPositiveVotes = votes

    def setNegativeVotes(self, votes: int):
        self.numberNegativeVotes = votes

    def setCreationDate(self, datestring: str):
        self.creationDate = datestring
