
import panel as pn
import holoviews as hv

def hvDyn(**kdimValues):
    """
    decorate to a function that takes some arguments and returns a holoviews plot
    run the function when defined
    For example: 
        @hvDyn(amp=[1,2,3,4])
        def plot(amp):
            xs = np.arange(-2*np.pi, np.pi*2, 0.01)
            return hv.Curve((xs, amp*np.sin(xs)))
    """
    def warpped_decorator(func):
        return hv.DynamicMap(func, kdims=list(kdimValues.keys())).redim.values(**kdimValues)
    return warpped_decorator

def serve(plot): 
    return pn.serve(plot)