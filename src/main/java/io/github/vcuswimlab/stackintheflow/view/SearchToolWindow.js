
var questionsList;
var numQuestions;

$(document).ready(function(){
    questionsList = new Array();
    numQuestions = 0;
});

$("#searchButton").click(function(){
    JavaBridge.searchButtonClicked($('#searchBar').val());
});

function Question(title, body, tags){
    this.title = title;
    this.body = body;
    this.tags = tags;
}

function getQuestion(title, body, tags){
    questionsList.push(new Question(title, body, tags));
    numQuestions++;
}

function displayQuestions(){
    JavaBridge.print("Displaying questions!");
    for(var i = 0; i < numQuestions; i++){
        $('#questionPlaceholder').append(questionsList[i].title);
        $('#questionPlaceholder').append(questionsList[i].body);
        for(var j = 0; j < questionsList[i].tags.length; j++){
            $('#questionPlaceholder').append(questionsList[i].tags[j].toString());
            $('#questionPlaceholder').append('<br />');
        }
        $('#questionPlaceholder').append('<hr />');
    }
}