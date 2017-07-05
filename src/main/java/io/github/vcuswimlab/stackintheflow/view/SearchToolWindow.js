
var questionsList;
var numQuestions;
var charCutoff;

$(document).ready(function(){
    questionsList = new Array();
    numQuestions = 0;
    charCutoff = 300;
});

$("#searchButton").click(function(){
    JavaBridge.searchButtonClicked($('#searchBox').val());
});

function Question(title, body, tags){
    this.title = title;
    this.body = body;
    this.tags = tags;
    this.codeTags = new Array();

    this.findCodeTags = function(){
        var lastStartIndex = 0;
        var lastEndIndex = 0;
        while(true){
            var start = this.body.indexOf("<pre>", lastStartIndex + 1);
            var end = this.body.indexOf("</pre>", lastEndIndex + 1);

            if(start != end && start != -1){
                this.codeTags.push(new Array(start, end));
                lastStartIndex = start;
                lastEndIndex = end;
            }
            else {
                break;
            }
        }
    }

    this.hasCodeTags = function(){
        return this.codeTags.length > 0;
    }

    this.lastCodeOpenTag = function(){
        JavaBridge.print("In last code open method");
        if(this.hasCodeTags()){
             JavaBridge.print("Has tags");
             return this.codeTags[this.codeTags.length - 1][0];
        }
        else {
            JavaBridge.print("Doesn't have tags");
            return -1;
        }
    }

    this.lastCodeCloseTag = function(){
        if(this.hasCodeTags()){
            return this.codeTags[this.codeTags.length - 1][1];
        }
        else {
            return -1;
        }
    }
}

function getQuestion(title, body, tags){
    questionsList.push(new Question(title, body, tags));;
    numQuestions++;
}

function displayQuestions(){
    JavaBridge.print("Displaying questions!");
    for(var i = 0; i < numQuestions; i++){
        JavaBridge.print("Question: " + i)
        var qTitle = $("<h3>").addClass('questionTitle').html(questionsList[i].title);
        $('#questions').append(qTitle);

        var qBody = $('<div>').addClass('questionBody').html(questionsList[i].body);
        $('#questions').append(qBody);

/*
        for(var j = 0; j < questionsList[i].tags.length; j++){
            $('#questions').append(questionsList[i].tags[j].toString());
            $('#questions').append('<br />');
        } */
        $('#questions').append('<hr />');
    }
}