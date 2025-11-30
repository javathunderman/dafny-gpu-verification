#include "clang/AST/AST.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/RawCommentList.h"

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("my-tool options");

class AssertionVisitor : public RecursiveASTVisitor<AssertionVisitor> {
 public:
  explicit AssertionVisitor(ASTContext *Context) : context(Context) {}
  bool VisitStmt(Stmt *stmt) {
        // stmt->printPretty(llvm::outs(),
        //                   nullptr,
        //                   context->getPrintingPolicy());
        return true;
    }
    bool VisitCallExpr(CallExpr *CE) {
        // Check if the expression is a kernel launch pattern
        if (CE->getNumArgs() >= 2) {
            Expr *FirstArg = CE->getArg(0);
            Expr *SecondArg = CE->getArg(1);
            if (FirstArg->getType().getAsString() == "dim3" && 
                SecondArg->getType().getAsString() == "dim3") {
                
                llvm::outs() << "Found CUDA kernel launch: "
                             << CE->getDirectCallee()->getNameInfo().getAsString() << "\n";
                SourceRange range = FirstArg->getSourceRange();
                if (range.isValid()) {
                    SourceManager &SM = context->getSourceManager();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM, context->getLangOpts());
                    llvm::outs() << "First Expression: " << text << "\n";
                }
                SourceRange range2 = SecondArg->getSourceRange();
                if (range2.isValid()) {
                    SourceManager &SM = context->getSourceManager();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range2), SM, context->getLangOpts());
                    llvm::outs() << "Second Expression: " << text << "\n";
                }
            }
        }
        return true;
    }

 private:
  ASTContext *context;
};

class AssertionConsumer : public clang::ASTConsumer {
 public:
  explicit AssertionConsumer(ASTContext *Context) : visitor_(Context) {}

    virtual void HandleTranslationUnit(clang::ASTContext& context) {
        visitor_.TraverseDecl(context.getTranslationUnitDecl());
        auto comments = context.Comments.getCommentsInFile(
            context.getSourceManager().getMainFileID());
        // llvm::outs() << context.Comments.empty() << "\n";
        if (!context.Comments.empty()) {
            for (auto it = comments->begin(); it != comments->end(); it++) {
                clang::RawComment* comment = it->second;
                std::string source = comment->getFormattedText(context.getSourceManager(),
                    context.getDiagnostics());
                llvm::outs() << source << "\n";
            }
        }
    }

 private:
  AssertionVisitor visitor_;
};

class AssertionFrontendAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<AssertionConsumer>(&Compiler.getASTContext());
  }
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();

  ClangTool tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  AssertionFrontendAction action;
  tool.run(newFrontendActionFactory<AssertionFrontendAction>().get());

  return 0;
}