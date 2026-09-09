%%
%% unimib-thesis.cls
%% Unofficial LaTeX class for Bachelor/Master/PhD theses
%% at Universita degli Studi di Milano - Bicocca.
%%
%% Class options:
%%   bachelor | master | phd   (degree level, default: master)
%%   italian  | english        (title-page language, default: italian)
%%   any other option (11pt, 12pt, ...) is passed through to 'report'
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{unimib-thesis}[2026/08/12 v1.0 Unimib Bachelor/Master/PhD thesis class]

%% ------------------------------------------------------------------
%% Options
%% ------------------------------------------------------------------
\newif\ifunimib@bachelor
\newif\ifunimib@master
\newif\ifunimib@phd
\newif\ifunimib@english

\unimib@mastertrue
\unimib@englishfalse

\DeclareOption{bachelor}{\unimib@bachelortrue\unimib@masterfalse\unimib@phdfalse}
\DeclareOption{master}{\unimib@mastertrue\unimib@bachelorfalse\unimib@phdfalse}
\DeclareOption{phd}{\unimib@phdtrue\unimib@bachelorfalse\unimib@masterfalse}
\DeclareOption{italian}{\unimib@englishfalse}
\DeclareOption{english}{\unimib@englishtrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\LoadClass[a4paper]{report}

%% ------------------------------------------------------------------
%% Packages
%% ------------------------------------------------------------------
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb,amsthm,amsfonts,latexsym}
\RequirePackage{bm}
\RequirePackage{setspace}
\RequirePackage{booktabs}
\RequirePackage{subcaption}
\RequirePackage[font=small,labelfont=bf]{caption}
\RequirePackage{titlesec}
\RequirePackage{geometry}
\RequirePackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}

%% ------------------------------------------------------------------
%% Page geometry and spacing
%% ------------------------------------------------------------------
\geometry{a4paper,tmargin=4cm,bmargin=3.5cm,lmargin=3cm,rmargin=3cm}

\titlespacing*{\section}{0pt}{20pt}{30pt}
\titlespacing*{\subsection}{0pt}{20pt}{15pt}

\setlength{\abovedisplayskip}{20pt}
\setlength{\belowdisplayskip}{20pt}

%% ------------------------------------------------------------------
%% Title-page data commands
%% ------------------------------------------------------------------
% Note: the internal \@unimib@... storage macros use the starred
% (non-\long) form of \newcommand/\renewcommand throughout, since their
% emptiness is tested with \ifx...\@empty in \maketitlepage below, and
% \ifx treats \long and non-\long macros as different even if both
% expand to nothing.
\newcommand*{\@unimib@university}{Universit\`a degli Studi di Milano - Bicocca}
\newcommand{\university}[1]{\renewcommand*{\@unimib@university}{#1}}

\newcommand*{\@unimib@department}{Dipartimento di Fisica ``Giuseppe Occhialini''}
\newcommand{\department}[1]{\renewcommand*{\@unimib@department}{#1}}

\newcommand*{\@unimib@coursename}{}
\newcommand{\coursename}[1]{\renewcommand*{\@unimib@coursename}{#1}}

\newcommand*{\@unimib@advisor}{}
\newcommand{\advisor}[1]{\renewcommand*{\@unimib@advisor}{#1}}

\newcommand*{\@unimib@coadvisor}{}
\newcommand{\coadvisor}[1]{\renewcommand*{\@unimib@coadvisor}{#1}}

\newcommand*{\@unimib@studentid}{}
\newcommand{\studentid}[1]{\renewcommand*{\@unimib@studentid}{#1}}

\newcommand*{\@unimib@academicyear}{}
\newcommand{\academicyear}[1]{\renewcommand*{\@unimib@academicyear}{#1}}

\newcommand*{\@unimib@logo}{logo_unimib.pdf}
\newcommand{\logo}[1]{\renewcommand*{\@unimib@logo}{#1}}

\newcommand*{\@unimib@logoscale}{2.5}
\newcommand{\logoscale}[1]{\renewcommand*{\@unimib@logoscale}{#1}}

% Optional overrides: some courses/departments use different wording
% (e.g. "prova finale" instead of "tesi"). If left empty, sensible
% defaults based on the bachelor/master/phd class option are used.
\newcommand*{\@unimib@thesistype}{}
\newcommand{\thesistype}[1]{\renewcommand*{\@unimib@thesistype}{#1}}

\newcommand*{\@unimib@submittedby}{}
\newcommand{\submittedby}[1]{\renewcommand*{\@unimib@submittedby}{#1}}

%% ------------------------------------------------------------------
%% Language- and degree-dependent labels
%% ------------------------------------------------------------------
\ifunimib@english
  \newcommand{\unimib@advisorlabel}{Supervisor}
  \newcommand{\unimib@coadvisorlabel}{Co-supervisor}
  \newcommand{\unimib@idlabel}{Student ID}
  \newcommand{\unimib@ayearlabel}{Academic Year}
  \ifunimib@bachelor
    \newcommand{\unimib@coursetype}{Bachelor's Degree Course in}
    \newcommand{\unimib@presentedby}{Bachelor's Thesis by}
  \fi
  \ifunimib@master
    \newcommand{\unimib@coursetype}{Master's Degree Course in}
    \newcommand{\unimib@presentedby}{Master's Thesis by}
  \fi
  \ifunimib@phd
    \newcommand{\unimib@coursetype}{PhD Programme in}
    \newcommand{\unimib@presentedby}{PhD Thesis by}
  \fi
\else
  \newcommand{\unimib@advisorlabel}{Relatore}
  \newcommand{\unimib@coadvisorlabel}{Correlatore}
  \newcommand{\unimib@idlabel}{Matricola}
  \newcommand{\unimib@ayearlabel}{Anno Accademico}
  \ifunimib@bachelor
    \newcommand{\unimib@coursetype}{Corso di Laurea in}
    \newcommand{\unimib@presentedby}{Relazione della prova finale di}
  \fi
  \ifunimib@master
    \newcommand{\unimib@coursetype}{Corso di Laurea Magistrale in}
    \newcommand{\unimib@presentedby}{Tesi di Laurea Magistrale di}
  \fi
  \ifunimib@phd
    \newcommand{\unimib@coursetype}{Dottorato di Ricerca in}
    \newcommand{\unimib@presentedby}{Tesi di Dottorato di}
  \fi
\fi

%% ------------------------------------------------------------------
%% Title page
%% ------------------------------------------------------------------
\newcommand{\maketitlepage}{%
  \begin{titlepage}
    \centering

    \begin{minipage}[t]{\textwidth}
      \centering
      \setstretch{1.12}
      {\Large\textsc{\@unimib@university}}\\
      \ifx\@unimib@department\@empty\else
        \large\textbf{\@unimib@department}\\
      \fi
      \textbf{%
        \ifx\@unimib@thesistype\@empty
          \unimib@coursetype\ \@unimib@coursename
        \else
          \@unimib@thesistype
        \fi
      }\\
      \par
    \end{minipage}

    \vspace{12mm}

    \begin{minipage}[t]{\textwidth}
      \centering
      \includegraphics[scale=\@unimib@logoscale]{\@unimib@logo}
    \end{minipage}

    \vspace{12mm}

    \begin{center}
      \Huge\textbf{\@title}
    \end{center}

    \vspace{22mm}

    \begin{flushleft}
      \ifx\@unimib@advisor\@empty\else
        {\large\textbf{\unimib@advisorlabel:} \@unimib@advisor}\\[2mm]
      \fi
      \ifx\@unimib@coadvisor\@empty\else
        {\large\textbf{\unimib@coadvisorlabel:} \@unimib@coadvisor}\\
      \fi
    \end{flushleft}

    \vspace{10mm}

    \begin{flushright}
      {\large\textbf{%
        \ifx\@unimib@submittedby\@empty
          \unimib@presentedby:
        \else
          \@unimib@submittedby:
        \fi
      }}\\
      \large{\@author}\\
      \ifx\@unimib@studentid\@empty\else
        \large{\unimib@idlabel\ \@unimib@studentid}
      \fi
    \end{flushright}

    \vspace{9mm}

    \begin{center}
      {\large\textbf{\unimib@ayearlabel\ \@unimib@academicyear}}
    \end{center}

    \restoregeometry
  \end{titlepage}
}