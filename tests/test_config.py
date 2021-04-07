import pytest
class NotAValidFileName(Exception):
    def __init__(self,message="Not a Valid File name"):
        self.message = message
        super().__init__(self.message)


def test_generic():
    filename ='train.csv'
    with pytest.raises(NotAValidFileName):
        if filename != 'test.csv':
            raise NotAValidFileName