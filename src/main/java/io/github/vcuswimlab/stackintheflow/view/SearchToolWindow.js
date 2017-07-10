
var questionsList;
var numQuestions;
var charCutoff;
var questionSections;

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
    questionSections = new Array();
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

    this.hasMoreContent = function(){
        return this.body.length > this.getShortenedContent().length;
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

/*
function displayQuestions(){
    JavaBridge.print("Displaying questions!");
    for(var i = 0; i < numQuestions; i++){
        //JavaBridge.print("Question: " + i);
        //JavaBridge.print(questionsList[i].body);

        var questionDiv = $("<div>").addClass('contentShortened');

        var qTitle = $("<h3>").addClass('questionTitle').html(questionsList[i].title);
        $(questionDiv).append(qTitle);

        var questionIndex = $('<span>').addClass('hidden').attr('id', 'questionIndex').html(i);

        var qBody = $('<div>').addClass('questionBody').html(questionsList[i].getShortenedContent());
        $(questionDiv).append(qBody);

        $(questionDiv).append(questionIndex);

        $('#questions').append(questionDiv);
        questionDivs.push(questionDiv);
        $('#questions').append('<hr />');
    }
} */


function displayQuestions(){
    for(var i = 0; i < numQuestions; i++){
        appendNewResultSkeleton(i);
        var questionSection = questionSections[i];
        $(questionSection).find(".searchResultTitle").html(questionsList[i].title);

        var questionBody = $(questionSection).find(".questionBody");
        $(questionBody).html(questionsList[i].getShortenedContent());

        if(questionsList[i].hasMoreContent()){
            var excerptController = $("<div>").addClass("excerptController").html("More");
            var lastChild = $(questionBody).children().last();
            if($(lastChild).is("PRE")){
                excerptController.removeClass('inlineExcerptController');
                excerptController.addClass('blockExcerptController');
                $(questionBody).append(excerptController);
            }
            else {
                excerptController.addClass('inlineExcerptController');
                excerptController.removeClass('blockExcerptController');
                $(lastChild).append(excerptController);
            }

            //$(questionBody).children().last().append(excerptController);
        }

        var questionTagsContainer = $(questionSection).find(".questionTags");
        var unorderedList = $("<ul>");

        for(var j = 0; j < questionsList[i].tags.length; j++){
            var tagItem = $("<li>").html(questionsList[i].tags[j].toString());
            $(unorderedList).append(tagItem);
        }
        $(questionTagsContainer).append(unorderedList);
    }
}

function appendNewResultSkeleton(i){
    var questionSection = $("<section>").addClass("searchResultItem");
    var questionBodyDiv = $("<div>").addClass("searchResultItemBody");
    var rowDiv = $("<div>").addClass("row");
    var questionBodyContentContainer = $("<div>").addClass("col-xs-12");
    var questionTitle = $("<h3>").addClass("searchResultTitle");
    var questionBodyContent = $("<div>").addClass("questionBody contentShortened");

    $(questionBodyContentContainer).append(questionTitle);
    $(questionBodyContentContainer).append(questionBodyContent);

    $(rowDiv).append(questionBodyContentContainer);

    $(questionBodyDiv).append(rowDiv);

    var questionIndex = $('<span>').addClass('hidden').attr('id', 'questionIndex').html(i);
    $(questionSection).append(questionIndex);
    $(questionSection).append(questionBodyDiv);

    var rowDiv2 = $("<div>").addClass("row");
    var tagsDiv = $('<div>').addClass("questionTags col-xs-12");

    $(rowDiv2).append(tagsDiv);

    $(questionSection).append(rowDiv2);

    $(questions).append(questionSection);
    questionSections.push(questionSection);
}

function generateListeners(){
    for(var i = 0; i < questionSections.length; i++){
        $(questionSections[i]).delegate(".excerptController", "click", function(e){
            JavaBridge.print("Click");
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            var questionBody = $(clickedSection).find('.questionBody');
            if($(this).html() == 'More'){
                $(questionBody).html(questionsList[index].body);
                $(this).html("Less");
            }
            else {
                $(questionBody).html(questionsList[index].getShortenedContent());
                $(this).html("More");
            }

            var lastChild = $(questionBody).children().last();
            if($(lastChild).is("PRE")){
                $(this).removeClass('inlineExcerptController');
                $(this).addClass('blockExcerptController');
                $(questionBody).append($(this));
            }
            else {
                $(this).addClass('inlineExcerptController');
                $(this).removeClass('blockExcerptController');
                $(lastChild).append($(this));
            }
        });

/*
        var excerptController = $(questionSections[i]).find(".excerptController");
        $(excerptController).click(function(){
            JavaBridge.print("Click");
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            var questionBody = $(clickedSection).find('.questionBody');
            if($(this).html() == 'More'){
                $(questionBody).html(questionsList[index].body);
                $(this).html("Less");
            }
            else {
                $(questionBody).html(questionsList[index].getShortenedContent());
                $(this).html("More");
            }
            $(questionBody).append($(this));
        }); */
        $(questionSections[i]).dblclick(function(){
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