# python script to hold utility functions used in the event detection project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from IPython.display import Markdown, display
import ipywidgets as widgets
from IPython.display import display, clear_output


def printmd(string):
    display(Markdown(string))


# python script to hold utility functions used in the event detection project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from IPython.display import Markdown, display
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import HBox

def printmd(string):
    display(Markdown(string))

def render_document_frontend(counter=[0,1],id='',title='',snippet='',body='',art='',lines=[],colors=['red','blue'],labels=[], metadata=None):
    '''
    Renders news articles and colors substrings within lines
    
    '''
    line_br = "<br>"
    document  = str(counter[1])+'/'+str(counter[0]) + line_br
    document+= 'id: '+str(id)+line_br
    if metadata is not None:
        document += metadata + line_br
        
    if not pd.isnull(title):
        document+= ("<h3> <center>"+title+" </center></h3>"+line_br)
        document+=line_br
    document+='<p>'
    if not pd.isnull(snippet):
        document+=snippet
    if not pd.isnull(body):
        document+=body
    document+='</p>'
    if len(art)>0:
        document+=('<p>'+art+'</p>')

    for i,line in enumerate(lines):
        document = re.sub(line,"<span style='color:"+colors[i]+"'><b>"+line+"<b>["+labels[i]+"]</span>",document)
    #print('test: ',document,len(document))
    return printmd(document)

class Frontend_docs:

    def __init__(self,sample, event_type,metadata_fields=None):
    
        self.i = 0
        self.sample = sample
        self.event_type = event_type    
        self.metadata_fields = metadata_fields
    
        keywords = ['opening','closing']           
        button_1 = widgets.Button(description="Next")
        button_2 = widgets.Button(description="Previous")
        button_3 = widgets.Button(description=event_type)
        button_4 = widgets.Button(description="no "+event_type)

        
        self.color = 'blue'

        box = HBox([button_2,button_1 ,button_3,button_4])
        
        display(box)
        
        self.out = widgets.Output()
        display(self.out)

        button_1.on_click(self.on_button_clicked_1)
        button_2.on_click(self.on_button_clicked_2)
        button_3.on_click(self.on_button_clicked_3)
        button_4.on_click(self.on_button_clicked_4)
        
        # initialize columns
        #if not self.event_type in self.sample.columns:
        self.sample[self.event_type] = 0
        
        #self.labels = np.zeros(self.sample.shape[0])

        #return self.labels
        
    def render(self,idx,len_sample):
        
        if self.metadata_fields is not None:
            metadata = ','.join([self.sample[field].iloc[idx] for field in self.metadata_fields])
            #print(metadata)
            render_document_frontend([len_sample,idx],self.sample['id'].iloc[idx],self.sample['title'].iloc[idx],
                        self.sample['snippet'].iloc[idx],self.sample['body'].iloc[idx],colors=[self.color]*10,metadata=metadata)
        else:

            render_document_frontend([len_sample,idx],self.sample['id'].iloc[idx],self.sample['title'].iloc[idx],
                        self.sample['snippet'].iloc[idx],self.sample['body'].iloc[idx],colors=[self.color]*10)

    def on_button_clicked_1(self,b):
        with self.out:
            clear_output()
            if ((self.i < len(self.sample)) and (self.i>=0)):
                self.render(self.i,len(self.sample))
                self.i+=1
            else:
                print('Done')
                clear_output()

    def on_button_clicked_2(self,b):
        with self.out:
            clear_output()
            if ((self.i < len(self.sample)) and (self.i>=0)):
                self.render(self.i,len(self.sample))
                self.i-=1
            else:
                print('Done')
                
    def on_button_clicked_3(self,b):
        self.sample.loc[self.sample.index[self.i-1],self.event_type] = 1
        
    def on_button_clicked_4(self,b):
        self.sample.loc[self.sample.index[self.i-1],self.event_type] = 0
        