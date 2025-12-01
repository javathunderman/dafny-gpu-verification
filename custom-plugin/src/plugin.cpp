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
#include "../include/dafny.hpp"
using namespace clang;
using namespace clang::tooling;
static llvm::cl::OptionCategory MyToolCategory("my-tool options");
// using ExprVariant = std::variant<clang::ImplicitCastExpr*, clang::BinaryOperator*>;
std::unordered_map<std::string, std::string> ind_constraints;

class AssertionVisitor : public RecursiveASTVisitor<AssertionVisitor> {
 public:
  explicit AssertionVisitor(ASTContext *Context) : context(Context) {}
  std::unordered_map<std::string, clang::Expr*> lexical_map;
  std::unordered_map<std::string, clang::FunctionDecl *> kernel_map;
  std::unordered_map<std::string, clang::VarDecl*> kernel_vars;
  FunctionDecl *curr_func;
  bool VisitStmt(Stmt *stmt) {
        #ifdef DEBUG_STMT
        stmt->printPretty(llvm::outs(),
                          nullptr,
                          context->getPrintingPolicy());
        #endif
        return true; 
    }
    bool VisitVarDecl(VarDecl *VD) {
        // Check if the variable is defined (not just declared)
        if (context->getSourceManager().getFileID(VD->getLocation()) == context->getSourceManager().getMainFileID() && VD->hasDefinition()) {
            if (VD->getType().getAsString() == "dim3") {
                #ifdef DEBUG_VAR_LOC
                llvm::outs() << "Variable defined: " << VD->getNameAsString() << "\n";
                llvm::outs() << "Definition located at: "
                            << VD->getLocation().printToString(context->getSourceManager())
                            << "\n";
                #endif
                lexical_map[VD->getNameAsString()] = VD->getInit();
            } else if (VD->getType()->isIntegerType()) {
                DeclContext *parentContext = VD->getDeclContext();
                if (clang::FunctionDecl *funcDecl = llvm::dyn_cast<clang::FunctionDecl>(parentContext)) {
                    if (kernel_map[funcDecl->getNameAsString()]) {
                        llvm::outs() << "Belongs to a kernel function " << funcDecl->getNameAsString() << " " << VD->getNameAsString() << "\n";
                        kernel_vars[VD->getNameAsString()] = VD;
                    }
                }
            }
        }
        return true;
    }
    
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *arrSubExpr) {
        Expr *indexExpr = arrSubExpr->getIdx();
        Expr *baseExpr = arrSubExpr->getBase();
        if (kernel_map[curr_func->getNameInfo().getName().getAsString()]) {
            if (clang::ImplicitCastExpr *baseExprI = llvm::dyn_cast<clang::ImplicitCastExpr>(baseExpr)) {
                // baseExprI->getSubExprAsWritten()->printPretty(llvm::outs(),
                //                 nullptr,
                //                 context->getPrintingPolicy());
                // llvm::outs() << "\n" << baseExprI->getSubExprAsWritten()->getStmtClassName() << "\nAttempt to print the parent, then indexExpr\n";
                SourceRange range_base = baseExprI->getSubExprAsWritten()->getSourceRange();
                SourceManager &SM = context->getSourceManager();
                llvm::StringRef base_text_ref = Lexer::getSourceText(CharSourceRange::getTokenRange(range_base), SM, context->getLangOpts());
                std::string base_text(base_text_ref.begin(), base_text_ref.end());
                if (clang::ImplicitCastExpr *indexExprI = llvm::dyn_cast<clang::ImplicitCastExpr>(indexExpr)) {
                    #ifdef DEBUG_ARR_SUBSCRIPT
                    llvm::outs()<< "Implicit cast expr for array subscript\n";
                    #endif
                    // indexExprI->getSubExprAsWritten()->printPretty(llvm::outs(),
                    //             nullptr,
                    //             context->getPrintingPolicy());
                    SourceRange range_ind = indexExprI->getSubExprAsWritten()->getSourceRange();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range_ind), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    ind_constraints[base_text] = stdStr;

                } else if (clang::BinaryOperator *indexExprB = llvm::dyn_cast<clang::BinaryOperator>(indexExpr)) {
                    #ifdef DEBUG_ARR_SUBSCRIPT
                    llvm::outs()<< "BinOp for array subscript\n";
                    #endif
                    // indexExprB->printPretty(llvm::outs(),
                    //             nullptr,
                    //             context->getPrintingPolicy());
                    SourceRange range_ind = indexExprB->getSourceRange();
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range_ind), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    ind_constraints[base_text] = stdStr;
                }
                llvm::outs() << "\n";
            }
        }
        return true;
    }
    bool VisitCallExpr(CallExpr *CE) {
        if (CE->getNumArgs() >= 2) {
            SourceManager &SM = context->getSourceManager();
            Expr *FirstArg = CE->getArg(0);
            Expr *SecondArg = CE->getArg(1);
            if (FirstArg->getType().getAsString() == "dim3" && 
                SecondArg->getType().getAsString() == "dim3") {
                
                // llvm::outs() << "Found CUDA kernel launch: "
                //              << CE->getDirectCallee()->getNameInfo().getAsString() << "\n";
                
                SourceRange range = FirstArg->getSourceRange();
                if (range.isValid()) {
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    if (lexical_map[stdStr]) {
                        llvm::outs() << "Grid dimensions are ";
                        lexical_map[stdStr]->printPretty(llvm::outs(),
                            nullptr,
                            context->getPrintingPolicy());
                        llvm::outs() << "\n";
                    }
                }
                SourceRange range2 = SecondArg->getSourceRange();
                if (range2.isValid()) {
                    llvm::StringRef text = Lexer::getSourceText(CharSourceRange::getTokenRange(range2), SM, context->getLangOpts());
                    std::string stdStr(text.begin(), text.end());
                    if (lexical_map[stdStr]) {
                        llvm::outs() << "Block dimensions are ";
                        lexical_map[stdStr]->printPretty(llvm::outs(),
                            nullptr,
                            context->getPrintingPolicy());
                        llvm::outs() << "\n";
                    }
                }
            }
        }
        return true;
    }
    bool VisitFunctionDecl(FunctionDecl *f) {
        curr_func = f;
        if (f->hasAttr<CUDAGlobalAttr>()) {
            kernel_map[f->getNameInfo().getName().getAsString()] = f;
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
  std::string dafnyCode = readFile("test_matmul.dfy");
  std::string modifiedCode = dafnyCode;
  for (const auto& pair : ind_constraints) {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        modifiedCode = modifyDafnyCode(modifiedCode, "var idx" + pair.first + " := ", pair.second);
    }
  writeFile("modified_output.dfy", modifiedCode);
  std::cout << "Modified Dafny code written to: modified_output.dfy" << std::endl;
  return 0;
}