import pytest

from figgie_gym.market.common import Symbol
from tests.mocks import MockClient, MockQueueOwner


@pytest.fixture
def client1() -> MockClient:
    return MockClient("C1")


@pytest.fixture
def client2() -> MockClient:
    return MockClient("C2")


@pytest.fixture
def client3() -> MockClient:
    return MockClient("C3")


@pytest.fixture
def queue_owner() -> MockQueueOwner:
    return MockQueueOwner()


@pytest.fixture
def symbol() -> Symbol:
    return Symbol(777)
