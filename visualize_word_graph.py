import networkx as nx
from nltk.corpus import wordnet as wn
from plotly.graph_objs import (Layout,
                               Scatter,
                               Figure,
                               Data,
                               Marker,
                               Margin,
                               XAxis,
                               Font,
                               YAxis)
import plotly.plotly as plotly


def traverse(graph, start, node, hypo=True):
    """transvers the graph"""
    graph.depth[node.name] = node.shortest_path_distance(start)
    if hypo:
        hypo = node.hyponyms()
    else:
        hypo = node.hypernyms()
    if len(hypo) == 0:
        return
    for child in hypo:
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)


def hyp_graph(start, hypo=True):
    """hyper graph"""
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start, hypo)
    return G


def visualize_word(word, hypo=True):
    """returns a network graph of word, hyponym,(or hypernym if hypo=False)"""
    syn = wn.synsets(word)
    return hyp_graph(syn[0], hypo)


def scatter_nodes(pos, labels=None, color=None, size=10, opacity=1):
    """function from https://plot.ly/ipython-notebooks/networks/""" 
    # pos is the dict of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None the Plotly default color is used
    # size is the size of the dots representing the nodes
    # opacity is a value between [0,1] defining the node color opacity
    L = len(pos)
    trace = Scatter(x=[], y=[],  mode='markers', marker=Marker(size=[]))
    for k in range(L):
        trace['x'].append(pos[k][0])
        trace['y'].append(pos[k][1])
    attrib = dict(name='', text=labels, hoverinfo='text', opacity=opacity)  # a dict of Plotly node attributes
    trace = dict(trace, **attrib)  # concatenate the dict trace and attrib
    trace['marker']['size'] = size
    return trace


def scatter_edges(G, pos, line_color=None, line_width=1):
    """function from https://plot.ly/ipython-notebooks/networks/""" 
    trace = Scatter(x=[], y=[], mode='lines')
    for edge in G.edges():
        trace['x'] += [pos[edge[0]][0], pos[edge[1]][0], None]
        trace['y'] += [pos[edge[0]][1], pos[edge[1]][1], None]
        trace['hoverinfo'] = 'none'
        trace['line']['width'] = line_width
        if line_color is not None:  # when it is None a default Plotly color is used
            trace['line']['color'] = line_color
    return trace


def draw_graph(word, hypernym=False):
    """Draw graph of word hyponym, or hypernym if hypernym=True,
    Displays in Jupyter notebook.
    Requires pre-installed plotly apikey"""
    G = visualize_word(word, not hypernym)
    nxpos = nx.fruchterman_reingold_layout(G)
    labels = []
    pos = []
    for k, v in nxpos.items():
        labels.append(str(k()))
        pos.append(v)
    trace1 = scatter_edges(G, nxpos)
    trace2 = scatter_nodes(pos, labels=labels)

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    layout = Layout(title='Graph for word "{}"'.format(word),
                    font=Font(),
                    showlegend=False,
                    autosize=True,
                    # width=width,
                    # height=height,
                    xaxis=XAxis(axis),
                    yaxis=YAxis(axis),
                    #margin=Margin(
                    #  l=40,
                    #  r=40,
                    #  b=85,
                    #  t=100,
                    #  pad=0,),
                    # plot_bgcolor='#EFECEA', #set background color
                    hovermode='closest')

    data = Data([trace1, trace2])

    fig = Figure(data=data, layout=layout)
    return plotly.iplot(fig, filename='networkx')
