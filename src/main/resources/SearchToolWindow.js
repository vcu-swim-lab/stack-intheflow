
var questionsList;
var numQuestions;
var charCutoff;
var questionSections;
var searchTags;
var uiSettings;
var searchMethod;

$(document).ready(function(){
    charCutoff = 300;
    searchTags = new SearchTags();
    uiSettings = new UISettings();
    searchMethod = "RELEVANCE";

    $('#searchBox').keydown(function(e) {
        if(e.keyCode == 9 && !e.shiftKey) {
            e.preventDefault();

            var words = $('#searchBox').val().split(" ");
            searchTags.add(words[words.length - 1]);
            words.splice(words.length - 1, 1);
            $('#searchBox').val(words);
            search();
        }
        $("#autoQueryIcon").addClass("hidden");
    });

    $("#searchTags").on("click", "li", function(e){
        searchTags.remove($(this).html());
        search();
    });

    $("#searchMethodsMenu").on("click", "li", function(e){
        $('#searchMethodsMenu').find(".selectedItem").removeClass("selectedItem");
        $(this).children().first().addClass("selectedItem");
        var dropdownVal = $(this).children().first().html().toLowerCase();
        if(dropdownVal == "relevance")
            dropdownVal = "RELEVANCE";
        else if(dropdownVal == "votes")
            dropdownVal = "VOTES";
        else if(dropdownVal == "newest")
            dropdownVal = "CREATION";
        else if(dropdownVal == "active")
            dropdownVal = "ACTIVITY";

        searchMethod = dropdownVal;
        search();
    });

    //Activate Tooltips
    $('[data-toggle="tooltip"]').tooltip();
});

function UISettings(){
    this.isDark = false;

    this.updateUI = function(){
        if(this.isDark){
            document.getElementById("defaultSheet").disabled = true;
            document.getElementById("darkSheet").disabled = false;
        }
        else {
            document.getElementById("defaultSheet").disabled = false;
            document.getElementById("darkSheet").disabled = true;
        }
    }
}

function updateUISettings(isDark){
    uiSettings.isDark = isDark;
    uiSettings.updateUI();
}

function SearchTags(){
    this.tags = new Array();

    this.add = function(newTag){
        if(!this.contains(newTag)){
            this.tags.push(newTag);
            this.updateUI();
        }
    }

    this.remove = function(tagToRemove){
        for(var i = 0; i < this.tags.length; i++){
            if(this.tags[i] === tagToRemove){
                this.tags.splice(i, 1);
            }
        }
        this.updateUI();
    }

    this.contains = function(tagToCheck){
        for(var i = 0; i < this.tags.length; i++){
            if(this.tags[i] == tagToCheck){
                return true;
            }
        }
        return false;
    }

    this.updateUI = function(){
        $('#searchTags').empty();
        for(var i = 0; i < this.tags.length; i++){
            var tag = $('<li>').html(this.tags[i]);
            $('#searchTags').append(tag);
        }

        if(this.tags.length > 0){
            $('#content').removeClass("contentStart").addClass("contentWithSearchTags");
        }
        else {
            $('#content').removeClass("contentWithSearchTags").addClass("contentStart");
        }

        JavaBridge.debugBreakpoint();
     }

    this.clear = function(){
        this.tags = new Array();
        this.updateUI();
    }

    this.getQuerySyntax = function(){
        var query = "";

        for(var i = 0; i < this.tags.length; i++){
            query += "[" + this.tags[i] + "]" + (i == this.tags.length - 1 ? "" : " ");
        }

        return query;
    }
}

function autoSearch(query, backoff){
    reset();
    $('#searchBox').val(query);
    JavaBridge.autoQuery(query, backoff);
    generateListeners();
    $("#autoQueryIcon").removeClass("hidden");
}

function errorSearch(firstMessage, secondMessage, backoff){
    reset();
    $('#searchBox').val(secondMessage);
    JavaBridge.autoQuery(secondMessage, backoff);
    if(numQuestions == 0){
        reset();
        $('#searchBox').val(firstMessage);
        JavaBridge.autoQuery(firstMessage, backoff);
    }
    generateListeners();
    $("#autoQueryIcon").removeClass("hidden");
}

function search(){
    reset();
    var query = $('#searchBox').val() + " " + searchTags.getQuerySyntax();
    JavaBridge.searchButtonClicked(query, searchMethod);
    generateListeners();
    $("#autoQueryIcon").addClass("hidden");
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
    $('html, body').animate({ scrollTop: 0 }, 'fast');
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

    uiSettings.updateUI();
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
        $(questionSections[i]).on("click", ".excerptController", function(e){
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

        $(questionSections[i]).on("click", ".questionTags li", function(e){
            searchTags.add($(this).html());
            search();
        });

        $(questionSections[i]).on("click", ".searchResultTitle", function(e){
            var clickedSection = $(this).closest('.searchResultItem');
            var index = $(clickedSection).find('#questionIndex').html();
            JavaBridge.openInBrowser(questionsList[index].link);
        });
    }

    $(document).on("click", "a", function(e){
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

$.fn.copyCSS = function (source) {
    var dom = $(source).get(0);
    var dest = {};
    var style, prop;
    if (window.getComputedStyle) {
        var camelize = function (a, b) {
                return b.toUpperCase();
        };
        if (style = window.getComputedStyle(dom, null)) {
            var camel, val;
            if (style.length) {
                for (var i = 0, l = style.length; i < l; i++) {
                    prop = style[i];
                    camel = prop.replace(/\-([a-z])/, camelize);
                    val = style.getPropertyValue(prop);
                    dest[camel] = val;
                }
            } else {
                for (prop in style) {
                    camel = prop.replace(/\-([a-z])/, camelize);
                    val = style.getPropertyValue(prop) || style[prop];
                    dest[camel] = val;
                }
            }
            return this.css(dest);
        }
    }
    if (style = dom.currentStyle) {
        for (prop in style) {
            dest[prop] = style[prop];
        }
        return this.css(dest);
    }
    if (style = dom.style) {
        for (prop in style) {
            if (typeof style[prop] != 'function') {
                dest[prop] = style[prop];
            }
        }
    }


    JavaBridge.print(JSON.stringify(dest));
};