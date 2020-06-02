#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import weakref
import inspect
from typing import TypeVar, Generic, Callable

T = TypeVar('T')

class Dispatcher(Generic[T]):
    """
    An object which broadcasts messages to subscribers in a publish/subscribe pattern.

    Instances of this class broadcast a single type of data upon request.  The request
    to broadcast is made by calling the publish() method.  Code elements that want to receive
    published data must subscribe to the instance, by calling the subscribe() method.
    Subscribers can be any callable which expects the published data to be passed as a
    parameter.

    You should use a separate Dispatcher instance for each type of data to be broadcast.
    The type of data passed to the publish() method should be known in advance so that 
    subscribers know what to expect.

    A Dispatcher keeps a weak reference to its subscribers.  This isn't enough to keep
    the subscriber from being garbage collected, so anything without another reference
    to it somewhere else will automatically be removed from the list of subscribers.

    Example:
    ---------
    def some_function(str):
        print("some_function called with", str)
    
    class Example():
        def foo(self, str):
            print("Example.foo called with", str)

    d = Dispatcher()
    d.subscribe(some_function)
    c = Example()
    d.subscribe(c.foo)
    d.publish("Hello")
    c = None
    l = lambda s : print("Referenced lambda says", s)
    d.subscribe(l)
    d.subscribe(lambda s : print("Inline lambda says", s))
    d.publish("Hello again")

    # Expected output:
    # ------------------
    # some_function called with Hello
    # Example.foo called with Hello
    # some_function called with Hello again
    # Referenced lambda says Hello again
    """

    def __init__(self):
        self._subscribers = []

    def subscribe(self, subscriber: Callable[[T], None]):
        """
        Indicate that a callable should be called when new data is published

        Parameters
        ----------
        subscriber: callable
            A callable that should be called when new data is published.  It will be called
            with a single argument, the data that is being published.
        """
        weakref_builder = weakref.WeakMethod if inspect.ismethod(subscriber) else weakref.ref
        self._subscribers.append(weakref_builder(subscriber))

    def publish(self, report: T):
        """
        Request that new data be sent to subscribers.
        
        In response to this method being called,  all of this instance's subscribers will
        be called, with the report being passed as a single argument to the callable subscriber.

        Parameters
        ----------
        report: object
           The data that is to be sent to subscribers.
        """
        deads = []
        for subscriber in self._subscribers:
            subscriber_callable = subscriber()
            if subscriber_callable is not None:
                subscriber_callable(report)
            else:
                deads.append(subscriber)
        # Trim dead references
        for dead in deads:
            self._subscribers.remove(dead)

