
from dataclasses import dataclass

from typing import List


@dataclass
class User:

    user_id : int
    comments : list


    def addComment(self, comment : str):
        self.comments.append(comment)
