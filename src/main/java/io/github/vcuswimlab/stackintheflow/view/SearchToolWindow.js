
var questionsList;
var numQuestions;
var charCutoff;
var questionDivs;

$(document).ready(function(){
    charCutoff = 300;
});

function search(){
    reset();
    JavaBridge.searchButtonClicked($('#searchBox').val());
    generateListeners();
    JavaBridge.addLinkListeners();
}

$("#searchButton").click(function(){
    search();
});

$(document).on('keypress', '#searchBox', function(e){
    if(e.which == 13){
        search();
    }
});

function reset(){
    $('#questions').empty();
    questionsList = new Array();
    questionDivs = new Array();
    numQuestions = 0;
}

function Question(title, body, tags, link){
    this.title = title;
    this.body = body;
    this.tags = tags;
    this.link = link;
    this.codeTags = new Array();
    this.htmlTags = new Array();

    this.findCodeTags = function(){
        var lastStartIndex = -1;
        var lastEndIndex = -1;
        while(true){
            var start = this.body.indexOf("<pre>", lastStartIndex + 1);
            var end = this.body.indexOf("</pre>", lastEndIndex + 1);

            if(start != end && start != -1){
                this.codeTags.push(new CodeTag(start, end));
                lastStartIndex = start;
                lastEndIndex = end;
            }
            else {
                break;
            }
        }
    }

    this.findHTMLTags = function(){
        var lastStartIndex = -1;
        var lastEndIndex = -1;
        while(true){
            var startMatch = this.body.regexIndexOf("<\\w[^>]*>", lastStartIndex + 1);
            var startIndex = parseInt(startMatch.index);
            var endMatch = this.body.regexIndexOf("<\\/[^>]+>", lastEndIndex + 1);
            var endIndex = parseInt(endMatch.index);

            if(startIndex != endIndex && startIndex != -1){
                this.htmlTags.push(new HTMLTag(startIndex, endIndex, endMatch.length));
                lastStartIndex = startIndex;
                lastEndIndex = endIndex;
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
        if(this.hasCodeTags()){
             return this.codeTags[this.codeTags.length - 1].open;
        }
        else {
            return -1;
        }
    }

    this.lastCodeCloseTag = function(){
        if(this.hasCodeTags()){
            return this.codeTags[this.codeTags.length - 1].close;
        }
        else {
            return -1;
        }
    }

    this.getCutoff = function(){
        var tolerance = 100;
        for(var i = this.codeTags.length - 1; i >= 0; i--){
            if(Math.abs(this.codeTags[i].close - charCutoff) < tolerance){
                return this.codeTags[i].close;
            }
            else if(Math.abs(this.codeTags[i].open - charCutoff) < tolerance){
                return this.codeTags[i].open - 1;
            }
            else if(this.codeTags[i].open < charCutoff && this.codeTags[i].close > charCutoff){
                return this.codeTags[i].open - 1;
            }
        }


        for(var i = 0; i < this.htmlTags.length; i++){
            if(charCutoff - this.htmlTags[i].open < this.htmlTags[i].length - 1 &&
                charCutoff - this.htmlTags[i].open > 0){
                return this.htmlTags[i].open - 1;
            }
            else if(charCutoff - this.htmlTags[i].close < this.htmlTags[i].length - 1 &&
                charCutoff - this.htmlTags[i].close > 0){
                return this.htmlTags[i].close + this.htmlTags[i].length;
            }
        }

        return charCutoff;
    }

    this.getShortenedContent = function(){
        return this.body.substring(0, this.getCutoff());
    }
}

function CodeTag(open, close){
    this.open = open;
    this.close = close;
}

function HTMLTag(open, close, length){
    this.open = open;
    this.close = close;
    this.length = length;
}

function getQuestion(title, body, tags, link){
    questionsList.push(new Question(title, body, tags, link));;
    questionsList[numQuestions].findCodeTags();
    questionsList[numQuestions].findHTMLTags();
    numQuestions++;
}

function displayQuestions(){
    JavaBridge.print("Displaying questions!");
    for(var i = 0; i < numQuestions; i++){
        //JavaBridge.print("Question: " + i);
        //JavaBridge.print(questionsList[i].body);
        var questionDiv = $("<div>").addClass('contentShortened');

        var qTitle = $("<h3>").addClass('questionTitle').html(questionsList[i].title);
        $(questionDiv).append(qTitle);

        var lastTag = charCutoff;
        if(questionsList[i].hasCodeTags()){
            lastTag = questionsList[i].lastCodeCloseTag();
        }

        var questionIndex = $('<span>').addClass('hidden').attr('id', 'questionIndex').html(i);

        var qBody = $('<div>').addClass('questionBody').html(questionsList[i].getShortenedContent());
        $(questionDiv).append(qBody);

        $(questionDiv).append(questionIndex);

        $('#questions').append(questionDiv);
        questionDivs.push(questionDiv);
        $('#questions').append('<hr />');
    }
}

function generateListeners(){
    for(var i = 0; i < questionDivs.length; i++){
        $(questionDivs[i]).click(function(){
            if($(this).hasClass('contentShortened')){
                $(this).removeClass('contentShortened');
                var index = $(this).find('#questionIndex').html();
                $(this).find('.questionBody').html(questionsList[index].body);
            }
            else {
                $(this).addClass('contentShortened');
                var index = $(this).find('#questionIndex').html();
                $(this).find('.questionBody').html(questionsList[index].getShortenedContent());
            }
        });

        JavaBridge.print("Adding dblClick listeners");
        $(questionDivs[i]).dblclick(function(){
            var index = $(this).find('#questionIndex').html();
            JavaBridge.openInBrowser(questionsList[index].link);
        });
    }

    $(document).delegate("a", "click", function(e){
        e.preventDefault();
        JavaBridge.openInBrowser($(this).attr('href'));
    });
}

function RegexMatch(result, index, length){
    this.result = result;
    this.index = index;
    this.length = length;
}

String.prototype.regexIndexOf = function(regex, fromIndex){
  var str = fromIndex ? this.substring(fromIndex) : this;
  var match = str.match(regex);

  return new RegexMatch(match ? match[0] : "", match ? str.indexOf(match[0]) + fromIndex : -1, match ? match[0].length : -1);
}