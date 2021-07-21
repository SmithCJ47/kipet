template_css_report = """

html {
  position: relative;
  min-height: 100%;
  font-size: 14px;
}

body {
  margin-top: 60px;
  margin-bottom: 60px;
  background: #3d4444;
}

.container {
  max-width: 960px;
  background: white;
  padding-bottom: 20px;
  margin: 0 auto;
  position: relative;
  border-radius: 16px;
}

.container-left {
   width: 56px;
   position: absolute;
   left: -80px;
   border-radius: 8px;
}

.container-right {
   width: 150px;
   position: absolute;
   left: 984px;
   border-radius: 8px;
}

.outer {
}

p {
  font-size: 14px;
}

.nav-link {
  color: black;
}

.info-block {
  margin-top: 40px;
  margin-bottom: 40px;
}

.section-block {
}

table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
  width: 100%;
  border-left: 6px solid #1e7f9d;
}

td, th {
  border: 2px solid #fff;
  text-align: left;
  padding: 8px;
  vertical-align: middle;
}

tr:nth-child(even) {
  background-color: #eee;
}
tr:nth-child(odd) {
  background-color: #f5f5f5;
}

h2 {
  color: black;
  background-color: white;
  border-radius: 0px;
  padding: 8px;
  padding-left:10px;
  border: 2px solid #FFDBDC;
  border-left: 6px solid #A90006;
}

th {
  background-color: #dcf4ff;
  color: black;
}

tr:hover {
  background-color: #ddd;
}

.container a.external:hover {
  text-decoration: none;
}

.container a.external {
  color: black;
  text-decoration: underline;
}

.container a.pagelink:hover {
  text-decoration: none;
}

.container a.pagelink {
  color: black;
  text-decoration: underline;
}

a.external::after {
    
    background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' fill='%23333' viewBox='0 0 16 16'> <path fill-rule='evenodd' d='M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5z'/> <path fill-rule='evenodd' d='M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0v-5z'/> </svg>") 0 0 no-repeat;
    background-size: auto;
    background-size: 16px;
    content: "";
    display: inline-block;
    height: 16px;
    margin-left: 3px;
    width: 16px;
}

a.file {
  margin: 10px;
  margin-bottom: 10px;
}

a.toggle {
  background-color: #216b82;
  border-color: #216b82;
  color: white;
  padding: 5px;
  border-radius: 5px;
  font-size: 12px;
  text-decoration: underline;
}

a.toggle:hover {
  background-color: #1e7f9d;
  border-color: #1e7f9d;
  text-decoration: none;
}

a.side-bar {
  color: black;
  background-color: white;
}

a.side-bar:hover {
  color: white;
  background-color: #1e7f9d;
}


a.side-bar-right {
  color: black;
  background-color: white;
}

a.side-bar-right:hover {
  color: white;
  background-color: #1e7f9d;
}


.big-equations {
  overflow: auto;
}

.nav-tabs .nav-link {
  background-color: white;
  margin-right: 5px;
  border-color: #ddd;
}

.nav-pills .nav-link {
  border-radius: 16px;
}

.nav-tabs .nav-link.active {
  background-color: #dcf4ff;
  font-weight: 600;
}

.nav-left {
  position: fixed;
  width: 40px;
  top: 50%;
  transform: translate(0%, -50%);

}

.nav-right {
  position: fixed;
  width: 130px;
  top: 50%;
  transform: translate(0%, -50%);

}

.side-bar {
  height: 56px;
  width: 56px;
  border-radius: 16px;
  margin-top: 6px;
  margin-bottom: 6px;
  transition: 0.25s;
}

.side-bar-right {
  height: 56px;
  width: 130px;
  border-radius: 16px;
  margin-top: 6px;
  margin-bottom: 6px;
  transition: 0.25s;
}

.side-bar-margin {
  margin-top: 28px;
}

.side-bar-item {
  padding-left: 20px;
}


.side-bar-item:active { 
  background-color: #ccc;
}

.border-top {
  margin-top: 20px;
}

.chart {
    border: 2px solid #ddd;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 10px;
    height: 550px;
    position: relative;
}

.modal-dialog {
    max-width: 960px;
        }
        
.modal-content {
    margin: auto;
        }
 
.modal-header {
    background-color: #dcf4ff;
    padding:16px 16px;
    color:black;
 
}
 
iframe {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
    height: 100%;
    width: 100%;
}

"""

template_css_prism = """

/* PrismJS 1.23.0
https://prismjs.com/download.html#themes=prism-okaidia&languages=python&plugins=line-numbers */
/**
 * okaidia theme for JavaScript, CSS and HTML
 * Loosely based on Monokai textmate theme by http://www.monokai.nl/
 * @author ocodia
 */

code[class*="language-"],
pre[class*="language-"] {
	color: #f8f8f2;
	background: none;
	text-shadow: 0 1px rgba(0, 0, 0, 0.3);
	font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
	font-size: 1em;
	text-align: left;
	white-space: pre;
	word-spacing: normal;
	word-break: normal;
	word-wrap: normal;
	line-height: 1.5;

	-moz-tab-size: 4;
	-o-tab-size: 4;
	tab-size: 4;

	-webkit-hyphens: none;
	-moz-hyphens: none;
	-ms-hyphens: none;
	hyphens: none;
}

/* Code blocks */
pre[class*="language-"] {
	padding: 1em;
	margin: .5em 0;
	overflow: auto;
	border-radius: 0.3em;
}

:not(pre) > code[class*="language-"],
pre[class*="language-"] {
	background: #272822;
}

/* Inline code */
:not(pre) > code[class*="language-"] {
	padding: .1em;
	border-radius: .3em;
	white-space: normal;
}

.token.comment,
.token.prolog,
.token.doctype,
.token.cdata {
	color: #8292a2;
}

.token.punctuation {
	color: #f8f8f2;
}

.token.namespace {
	opacity: .7;
}

.token.property,
.token.tag,
.token.constant,
.token.symbol,
.token.deleted {
	color: #f92672;
}

.token.boolean {
	color: #ae81ff;
}

.token.number {
	color: #ff1e00;
}

.token.selector,
.token.attr-name,
.token.string,
.token.char,
.token.builtin,
.token.inserted {
	color: #a6e22e;
}

.token.operator,
.token.entity,
.token.url,
.language-css .token.string,
.style .token.string,
.token.variable {
	color: #f8f8f2;
}

.token.atrule,
.token.attr-value,
.token.function,
.token.class-name {
	color: #e6db74;
}

.token.keyword {
	color: #66d9ef;
}

.token.regex,
.token.important {
	color: #fd971f;
}

.token.important,
.token.bold {
	font-weight: bold;
}
.token.italic {
	font-style: italic;
}

.token.entity {
	cursor: help;
}

pre[class*="language-"].line-numbers {
	position: relative;
	padding-left: 4.8em;
	counter-reset: linenumber;
}

pre[class*="language-"].line-numbers > code {
	position: relative;
	white-space: inherit;
}

.line-numbers .line-numbers-rows {
	position: absolute;
	pointer-events: none;
	top: 0;
	font-size: 100%;
	left: -4.8em;
	width: 4em; /* works for line-numbers below 1000 lines */
	letter-spacing: -1px;
	border-right: 1px solid #999;

	-webkit-user-select: none;
	-moz-user-select: none;
	-ms-user-select: none;
	user-select: none;

}

	.line-numbers-rows > span {
		display: block;
		counter-increment: linenumber;
	}

		.line-numbers-rows > span:before {
			content: counter(linenumber);
			color: #999;
			display: block;
			padding-right: 0.8em;
			text-align: right;
		}

div.code-toolbar {
	position: relative;
}

div.code-toolbar > .toolbar {
	position: absolute;
	top: .3em;
	right: .2em;
	transition: opacity 0.3s ease-in-out;
	opacity: 0;
}

div.code-toolbar:hover > .toolbar {
	opacity: 1;
}

/* Separate line b/c rules are thrown out if selector is invalid.
   IE11 and old Edge versions don't support :focus-within. */
div.code-toolbar:focus-within > .toolbar {
	opacity: 1;
}

div.code-toolbar > .toolbar .toolbar-item {
	display: inline-block;
}

div.code-toolbar > .toolbar a {
	cursor: pointer;
}

div.code-toolbar > .toolbar button {
	background: none;
	border: 0;
	color: inherit;
	font: inherit;
	line-height: normal;
	overflow: visible;
	padding: 0;
	-webkit-user-select: none; /* for button */
	-moz-user-select: none;
	-ms-user-select: none;
}

div.code-toolbar > .toolbar a,
div.code-toolbar > .toolbar button,
div.code-toolbar > .toolbar span {
	color: #bbb;
	font-size: .8em;
	padding: 0 .5em;
	background: #f5f2f0;
	background: rgba(224, 224, 224, 0.2);
	box-shadow: 0 2px 0 0 rgba(0,0,0,0.2);
	border-radius: .5em;
}

div.code-toolbar > .toolbar a:hover,
div.code-toolbar > .toolbar a:focus,
div.code-toolbar > .toolbar button:hover,
div.code-toolbar > .toolbar button:focus,
div.code-toolbar > .toolbar span:hover,
div.code-toolbar > .toolbar span:focus {
	color: inherit;
	text-decoration: none;
}

"""

template_css = template_css_report + template_css_prism